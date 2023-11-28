import cv2
import numpy as np
from numpy import ndarray

# from sklearn._typing import ArrayLike, MatrixLike
from tqdm import tqdm
from skimage import feature
from sklearn.metrics import jaccard_score


from sklearn.base import TransformerMixin, BaseEstimator


class CoringDetector(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        metric: str = "f1_score",
        canny_sigma: float = 2.7,
        N1: int = 19,
        N2: int = 9,
        N3: int = 5,
        min_area: int = 20,
        max_area: int = 400,
        min_round_ratio: float = 0.5,
        max_round_ratio: float = 2.0,
    ) -> None:
        self.metric = metric
        self.canny_sigma = canny_sigma
        self.N1 = N1
        self.N2 = N2
        self.N3 = N3
        self.min_area = min_area
        self.max_area = max_area
        self.min_round_ratio = min_round_ratio
        self.max_round_ratio = max_round_ratio

        self.all_transforms = {}

    def apply_transforms(self, X: np.array):
        """
        Aplica serie de transformações para encontrar e fechar os contornos
        que representam os boreholes.
        """
        input_images = X.copy()
        self.all_transforms["input"] = input_images

        # Calcular bordas com canny
        images = [
            feature.canny(img, sigma=self.canny_sigma) * 255.0 for img in input_images
        ]
        self.all_transforms["canny"] = images

        # Fechar bordas encontradas com fechamento (para criar blobs)
        kernel1 = np.ones((self.N1, self.N1), np.uint8)
        images = [cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel1) for img in images]
        self.all_transforms["closing"] = images

        # Aplicando abertura com kernel vertical para remover ruidos horizontais
        # kernel2 = np.ones((self.N2, 1), np.uint8)
        # images = [cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel2) for img in images]
        # self.all_transforms["open"] = images

        # Aplicando dilatação para juntar ruidos proximos em grandes regiões, pode facilitar
        # na filtragem por área, mas talvez não seja necessario.
        # kernel3 = np.ones((self.N3, self.N3), np.uint8)
        # images = [cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel3) for img in images]
        # self.all_transforms["dialate"] = images

        self.all_transforms["output"] = images

        return images

    @staticmethod
    def find_contours(X: list):
        """
        Usa OpenCV para encotrar os blobs
        das images pre processadas.
        """

        images = X.copy()
        converted_images = []
        for image in np.array(images):
            # Ensure that the image is in uint8 format (CV_8U)
            if image.max() == 1:
                image *= 255
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
            # Convert to CV_8UC1 format
            if len(image.shape) > 2:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            converted_images.append(image)

        contours = []
        for img in converted_images:
            contours_list, _ = cv2.findContours(
                img, cv2.CHAIN_APPROX_SIMPLE, cv2.CHAIN_APPROX_NONE
            )
            contours.append(contours_list)

        return contours

    def apply_thresholds(self, contours: np.array):
        """
        Usa thresholds para filtrar contornos com base
        em algumas propriedades.
        """

        filtered_contours = []
        for contours_list in contours:
            new_contours_list = []
            for cntr in contours_list:
                area = cv2.contourArea(cntr)
                # contour_areas.append(area)

                arclength = cv2.arcLength(cntr, True)
                # arclengths.append(arclength)

                round_ratio = 4 * np.pi * area / (arclength**2)
                # roundness.append(round_ratio)

                if (
                    (round_ratio < self.max_round_ratio)
                    and (round_ratio > self.min_round_ratio)
                    and (area > self.min_area)
                    and (area < self.max_area)
                ):
                    new_contours_list.append(cntr)

            filtered_contours.append(new_contours_list)

        return filtered_contours

    def get_filtered_blobs(self, filtered_contours: list):
        """
        Usa os contornos filtrados para gerar a imges finais
        que representam os boreholes.
        """

        N_images = len(self.input_images)
        input_images_shape = (N_images, *(self.input_images[0].shape))
        masks = np.zeros(input_images_shape, dtype="uint8")

        for i, image in enumerate(masks):
            img_contours = filtered_contours[i]
            cv2.drawContours(
                image, img_contours, -1, (255, 255, 255), cv2.FILLED
            )  # -1 means draw all contours, (0, 255, 0) is the color, 2 is the thickness

        return masks

    @staticmethod
    def get_centroids(filtered_contours: list):
        centroids = []
        for contour_list in filtered_contours:
            centroids_list = []
            for cntr in contour_list:
                M = cv2.moments(cntr)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centroid = np.array([cx, cy])
                    centroids_list.append(centroid)

            centroids.append(centroids_list)

        return centroids

    def fit_transform(self, X: np.array) -> ndarray:
        transformed_images = self.apply_transforms(X)
        contours = self.find_contours(transformed_images)
        filtered_contours = self.apply_thresholds(contours)
        centroids_pred = self.get_centroids(filtered_contours)

        return centroids_pred

    def fit(self, X_train, y_train, **fit_params):
        return self

    def score(self, X, y):
        threshold = 20
        predicted_centroids = self.fit_transform(X)
        true_centroids = y.copy()

        try:
            assert len(true_centroids) == len(predicted_centroids)
        except Exception:
            print(
                f"true_centroids and predicted_centroids have incompatible lengths: {len(true_centroids)} and {len(predicted_centroids)}."
            )

        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.precision = 0
        self.recall = 0
        self.f1_score = 0

        for i in range(len(true_centroids)):
            for pred_centroid in predicted_centroids[i]:
                match_found = False
                for true_centroid in true_centroids[i]:
                    distance = np.linalg.norm(pred_centroid - true_centroid)
                    if distance < threshold:
                        match_found = True
                        break
                if match_found:
                    self.true_positives += 1
                else:
                    self.false_positives += 1

        self.false_negatives = len(true_centroids) - self.true_positives

        if self.true_positives + self.false_positives > 0:
            self.precision = self.true_positives / (
                self.true_positives + self.false_positives
            )

        if (self.true_positives + self.false_negatives) > 0:
            self.recall = self.true_positives / (
                self.true_positives + self.false_negatives
            )

        if (self.precision + self.recall) > 0:
            self.f1_score = (
                2 * (self.precision * self.recall) / (self.precision + self.recall)
            )

        self.true_positives_rate = self.true_positives / len(true_centroids)
        self.false_positives_rate = self.false_positives / len(true_centroids)
        self.false_negatives_rate = self.false_negatives / len(true_centroids)

        if self.metric == "precision":
            return self.precision
        elif self.metric == "recall":
            return self.recall
        elif self.metric == "f1_score":
            return self.f1_score
        elif self.metric == "TP":
            return self.true_positives_rate
        elif self.metric == "FP":
            return self.false_positives_rate
        elif self.metric == "FN":
            return self.false_negatives_rate
        else:
            raise Exception(
                f"{self.metric} is not valid. Valid metrics are 'precision', 'recall', 'f1_socore'"
            )

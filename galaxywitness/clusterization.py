import numpy as np


class Clusterization:
    """
    Class for handling clusterization of a point cloud

    :param n_clusters: number of clusters.
    :type n_clusters: int

    :param clusters: collection of clusters, each cluster contains 3d coordinates of its elements
    and (optional) weights of its elements. Order of variables: [coord_1, coord_2, coord_3, weight].
    :type clusters: ndarray (structured or homogeneous), Iterable, dict, or DataFrame

    :param centers_of_mass_computed: flag is set to False unless centers of mass were computed for clusters.
    :type centers_of_mass_computed: bool

    :param centers_of_mass: collection of computed centers of mass,
    info contained about each center: [coord_1, coord_2, coord_3, weight]
    :type: centers_of_mass: list()

    """
    __slots__ = [
        'n_clusters',
        'clusters',
        'centers_of_mass_computed',
        'centers_of_mass'
    ]

    def __init__(self, n_clusters, clusters):
        self.n_clusters = n_clusters
        self.clusters = np.array(clusters)
        self.centers_of_mass_computed = False
        self.centers_of_mass = list()

    def center_of_mass(self):
        for cluster in self.clusters:
            weight = sum(cluster[:, 3])
            av_x = sum(cluster[:, 0] * cluster[:, 3]) / weight
            av_y = sum(cluster[:, 1] * cluster[:, 3]) / weight
            av_z = sum(cluster[:, 2] * cluster[:, 3]) / weight
            self.centers_of_mass.append([av_x, av_y, av_z, weight])
        self.centers_of_mass_computed = True
        return self.centers_of_mass

    def compare_clusterization(self, other):
        if self.n_clusters != other.n_clusters:
            raise Exception("Cannot compare clusterizations with different sizes")
        if not self.centers_of_mass_computed:
            self.center_of_mass()
        if not other.centers_of_mass_computed:
            other.center_of_mass()

        def center_of_mass_diff(c1, c2):
            sum_square = (c1[0] - c2[0])**2 + (c1[1] - c2[1])**2 + (c1[2] - c2[2])**2
            dist = np.sqrt(sum_square)
            weight_diff = abs(c1[3] - c2[3])  # improve weight_diff metric
            return dist, weight_diff

        cm_dist_diff_sum = 0
        cm_weight_diff = 0
        # Greedy approach, O(n^2) complexity, the result depends on the order of centers_of_mass traversal.
        # To optimise with Voronoi diagrams?
        unmatched_c2_centers = other.centers_of_mass
        for center, idx in zip(self.centers_of_mass, range(self.n_clusters)):
            min_diff, min_diff_jdx = np.inf, 0
            min_weight_diff = 0
            for center_other, jdx in zip(unmatched_c2_centers, range(len(unmatched_c2_centers))):
                dist_diff, weight_diff = center_of_mass_diff(center, center_other)
                if dist_diff < min_diff:
                    min_diff = dist_diff
                    min_weight_diff = weight_diff
                    min_diff_jdx = jdx
            cm_dist_diff_sum += min_diff
            cm_weight_diff += min_weight_diff
            unmatched_c2_centers.pop(min_diff_jdx)
        return cm_dist_diff_sum + cm_weight_diff

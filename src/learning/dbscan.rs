//! DBSCAN Clustering
//!
//! *Note: This module is likely to change dramatically in the future and
//! should be treated as experimental.*
//!
//! Provides an implementaton of DBSCAN clustering. The model
//! also implements a `predict` function which uses nearest neighbours
//! to classify the points. To utilize this function you must use
//! `self.set_predictive(true)` before training the model.
//!
//! The algorithm works by specifying `eps` and `min_points` parameters.
//! The `eps` parameter controls how close together points must be to be
//! placed in the same cluster. The `min_points` parameter controls how many
//! points must be within distance `eps` of eachother to be considered a cluster.
//!
//! If a point is not within distance `eps` of a cluster it will be classified
//! as noise. This means that it will be set to `None` in the clusters `Vector`.
//!
//! # Examples
//!
//! ```
//! use rusty_machine::learning::dbscan::DBSCAN;
//! use rusty_machine::learning::UnSupModel;
//! use rusty_machine::linalg::Matrix;
//!
//! let inputs = Matrix::new(6, 2, vec![1.0, 2.0,
//!                                     1.1, 2.2,
//!                                     0.9, 1.9,
//!                                     1.0, 2.1,
//!                                     -2.0, 3.0,
//!                                     -2.2, 3.1]);
//!
//! let mut model = DBSCAN::new(&inputs, 0.5, 2);
//!
//! let clustering = model.clusters();
//! ```

use learning::error::{Error, ErrorKind};
use learning::{LearningResult, UnSupModel};

use itertools::Itertools;
use kdtree::distance::squared_euclidean;
use kdtree::KdTree;
use linalg::{BaseMatrix, Matrix, Vector};
use rulinalg::matrix::Row;
use rulinalg::utils;

/// DBSCAN Model
///
/// Implements clustering using the DBSCAN algorithm
/// via the `UnSupModel` trait.
#[derive(Debug)]
pub struct DBSCAN {
    eps: f64,
    min_points: usize,
    clusters: Vec<Option<usize>>,
    _visited: Vec<bool>,
}

/// Constructs a non-predictive DBSCAN model with the
/// following parameters:
///
/// - `eps` : `0.5`
/// - `min_points` : `5`
impl Default for DBSCAN {
    fn default() -> DBSCAN {
        DBSCAN {
            eps: 0.5,
            min_points: 5,
            clusters: vec![],
            _visited: vec![],
        }
    }
}

impl UnSupModel<Matrix<f64>, Vector<Option<usize>>> for DBSCAN {
    /// Train the classifier using input data.
    fn train(&mut self, inputs: &Matrix<f64>) -> LearningResult<()> {
        let mut cluster = 0;
        let mut neighbours = Vec::with_capacity(inputs.rows());
        let mut sub_neighbours = Vec::with_capacity(inputs.rows());

        let kdt = Self::kdtree(inputs);

        for (idx, point) in inputs.row_iter().enumerate() {
            let visited = self._visited[idx];

            idx;
            point;
            visited;
            if !visited {
                self._visited[idx] = true;

                neighbours.clear();
                self.region_query(point, inputs, &mut neighbours, &kdt);
                neighbours.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                neighbours.dedup();

                if neighbours.len() >= self.min_points {
                    self.expand_cluster(inputs, idx, &mut neighbours, &mut sub_neighbours, cluster, &kdt);
                    cluster += 1;
                }
            }
        }

        Ok(())
    }

    fn predict(&self, inputs: &Matrix<f64>) -> LearningResult<Vector<Option<usize>>> {
        panic!("removed");
    }
}

impl DBSCAN {
    /// Create a new DBSCAN model with a given
    /// distance episilon and minimum points per cluster.
    pub fn new(inputs: &Matrix<f64>, eps: f64, min_points: usize) -> DBSCAN {
        assert!(eps > 0f64, "The model epsilon must be positive.");

        let mut dbscan = DBSCAN {
            eps: eps,
            min_points: min_points,
            clusters: vec![None; inputs.rows()],
            _visited: vec![false; inputs.rows()],
        };
        dbscan.train(inputs).unwrap();
        dbscan
    }

    /// Clusters slice
    pub fn clusters<'a>(&'a self) -> &'a [Option<usize>] {
        &self.clusters
    }

    fn expand_cluster<'a>(&mut self, inputs: &Matrix<f64>, mut point_idx: usize, neighbours: &mut Vec<usize>, sub_neighbours: &mut Vec<usize>, cluster: usize, kdt: &KdTree<f64, usize, &'a [f64]>) {
        debug_assert!(point_idx < inputs.rows(), "Point index too large for inputs");
        self.clusters[point_idx] = Some(cluster);

        while let Some(data_point_idx) = neighbours.pop() {
            debug_assert!(data_point_idx < inputs.rows(), "Data point index too large for inputs");
            debug_assert!(neighbours.iter().all(|x| *x < inputs.rows()), "Neighbour indices too large for inputs");

            self.clusters[point_idx] = Some(cluster);
            let visited = self._visited[data_point_idx];
            if !visited {
                self._visited[data_point_idx] = true;
                let data_point_row = unsafe { inputs.row_unchecked(data_point_idx) };

                sub_neighbours.clear();
                self.region_query(data_point_row, inputs, sub_neighbours, kdt);

                if sub_neighbours.len() >= self.min_points {
                    point_idx = data_point_idx;
                    neighbours.extend_from_slice(&sub_neighbours);
                    neighbours.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                    neighbours.dedup();
                }
            }
        }
    }

    fn region_query<'a>(&self, point: Row<f64>, inputs: &Matrix<f64>, neighbours: &mut Vec<usize>, kdt: &KdTree<f64, usize, &'a [f64]>) {
        debug_assert!(point.cols() == inputs.cols(), "point must be of same dimension as inputs");
        for (pnt, idx) in kdt.within(point.raw_slice(), self.eps.powi(2), &squared_euclidean).expect("KdTree error checking point") {
            neighbours.push(*idx);
        }
    }

    fn kdtree<'a>(inputs: &'a Matrix<f64>) -> KdTree<f64, usize, &'a [f64]> {
        let mut kdt = KdTree::new(inputs.cols());
        for (idx, row) in inputs.row_iter().enumerate() {
            kdt.add(row.raw_slice(), idx);
        }
        kdt
    }

    /// Predict the clustering of new data given the data set the model was trained on.
    pub fn predict(&self, cluster_data: &Matrix<f64>, new_data: &Matrix<f64>) -> LearningResult<Vec<ClusterPrediction>> {
        let mut neighbours = Vec::with_capacity(cluster_data.rows());
        let kdt = Self::kdtree(cluster_data);
        Ok(new_data
            .row_iter()
            .map(|point| {
                neighbours.clear();
                self.region_query(point, cluster_data, &mut neighbours, &kdt);
                let mut clusters = neighbours.iter().map(|idx| self.clusters[*idx]).unique().collect::<Vec<Option<usize>>>();
                if neighbours.len() >= self.min_points - 1 {
                    ClusterPrediction::Core(clusters)
                } else if clusters.iter().any(|c| c.is_some()) {
                    ClusterPrediction::Border(clusters)
                } else {
                    ClusterPrediction::Noise
                }
            })
            .collect::<Vec<ClusterPrediction>>())
    }
}

/// Predicted cluster output
#[derive(Debug, Clone, PartialEq)]
pub enum ClusterPrediction {
    /// Point would be a core member of at least one cluster.
    /// Multiple values indicates the new point would trigger a merging of clusters with each other
    /// or with noise points.
    Core(Vec<Option<usize>>),
    /// Point would not be a core member but is within eps distance of at least one cluster.
    Border(Vec<Option<usize>>),
    /// Point is outside eps distance of all clusters.
    Noise,
}

#[cfg(test)]
mod tests {
    use super::DBSCAN;
    use linalg::{BaseMatrix, Matrix};

    #[test]
    fn test_region_query() {
        let inputs = Matrix::new(3, 2, vec![1.0, 1.0, 1.1, 1.9, 3.0, 3.0]);
        let model = DBSCAN::new(&inputs, 1.0, 3);
        let kdt = DBSCAN::kdtree(&inputs);

        let m = matrix![1.0, 1.0];
        let row = m.row(0);
        let mut neighbours = vec![];
        model.region_query(row, &inputs, &mut neighbours, &kdt);

        assert!(neighbours.len() == 2);
    }

    #[test]
    fn test_region_query_small_eps() {
        let inputs = Matrix::new(3, 2, vec![1.0, 1.0, 1.1, 1.9, 1.1, 1.1]);
        let model = DBSCAN::new(&inputs, 0.01, 3);
        let kdt = DBSCAN::kdtree(&inputs);

        let m = matrix![1.0, 1.0];
        let row = m.row(0);
        let mut neighbours = vec![];
        model.region_query(row, &inputs, &mut neighbours, &kdt);

        assert!(neighbours.len() == 1);
    }
}

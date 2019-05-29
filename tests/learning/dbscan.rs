use rm::linalg::Matrix;

use rm::learning::dbscan::{ClusterPredictions, DBSCAN};
use rm::learning::UnSupModel;

extern crate rand;
use self::rand::{thread_rng, Rng};

#[test]
#[ignore]
fn test_does_not_overflow() {
    let mut rng = thread_rng();
    let n = 100_000;
    let d = 2;
    let inputs = Matrix::new(n, d, rng.gen_iter::<f64>().take(n * d).collect::<Vec<f64>>());

    let mut model = DBSCAN::new(&inputs, 0.5, 2);

    let clustering = dbg!(model.clusters());
}

#[test]
fn test_basic_clusters() {
    let inputs = Matrix::new(8, 2, vec![1.0, 2.0, 1.1, 2.2, 0.9, 1.9, 1.0, 2.1, -2.0, 3.0, -2.2, 3.1, -1.0, -2.0, -2.0, -1.0]);

    let mut model = DBSCAN::new(&inputs, 0.5, 2);

    let clustering = dbg!(model.clusters());

    assert!(clustering.iter().take(4).all(|x| *x == Some(0)));
    assert!(clustering.iter().skip(4).take(2).all(|x| *x == Some(1)));
    assert!(clustering.iter().skip(6).all(|x| *x == None));
}

#[test]
fn test_border_points() {
    let inputs = Matrix::new(5, 1, vec![1.55, 2.0, 2.1, 2.2, 2.65]);

    let mut model = DBSCAN::new(&inputs, 0.5, 3);

    let clustering = dbg!(model.clusters());

    assert!(clustering.iter().take(1).all(|x| *x == None));
    assert!(clustering.iter().skip(1).take(3).all(|x| *x == Some(0)));
    assert!(clustering.iter().skip(4).all(|x| *x == None));
}

#[test]
fn test_basic_prediction() {
    let inputs = Matrix::new(6, 2, vec![1.0, 2.0, 1.1, 2.2, 0.9, 1.9, 1.0, 2.1, -2.0, 3.0, -2.2, 3.1]);

    let mut model = DBSCAN::new(&inputs, 0.5, 2);

    let new_points = Matrix::new(2, 2, vec![1.0, 2.0, 4.0, 4.0]);

    let classes = model.predict(&inputs, &new_points).unwrap();

    if let ClusterPredictions::Core(c0) = classes.get(0).unwrap() {
        assert!(c0.iter().any(|c| *c == Some(0)));
    } else {
        panic!("{:?}", classes[0]);
    }
    assert!(classes[1] == ClusterPredictions::Noise);
}

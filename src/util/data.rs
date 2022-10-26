use image::{io::Reader as ImageReader, Rgb};
use image::Rgb32FImage;
use rand::seq::SliceRandom;
use std::fs;
use std::path::Path;
use nshare::ToNdarray3;
use ndarray::{Ix, Axis};

use super::dtypes::*;

pub enum InputType {
    Array3F,
    // Video,
    // Text,
    // Num,
    //
}
#[derive(Debug)]
pub enum DataLoadingError {
    EmptyDir,
    UnexpSubdir,
}

pub struct Dataset {
    pub data_dir: String,
    data_paths: Vec<String>,

    pub input_type: InputType,
}

fn is_img_path(p: &Path) -> bool{
    match p.extension() {
        Some(pp) => {
            match pp.to_str() {
                Some(ppp) => {
                    match ppp {
                        "jpg" | "JPG" | "png" => true,
                        _ => false
                    }
                },
                None => false
            }
        },
        None => false 
    }
}

impl Dataset {
    pub fn new(data_dir: &str) -> Result<Dataset, DataLoadingError> {
        let mut pv: Vec<String> = Vec::new();

        let dir =
            fs::read_dir(data_dir).expect(format!("Could not load path: {}", data_dir).as_str());

        for item in dir {
            let pth = item.unwrap().path();

            if pth.is_file() {
                pv.push(pth.to_str().unwrap().to_string());
            }
        }

        if pv.is_empty() {
            return Err(DataLoadingError::EmptyDir);
        }

        Ok(Dataset {
            data_dir: data_dir.to_string(),
            data_paths: pv,

            input_type: InputType::Array3F,
        })
    }

    pub fn batch_paths(&self, batch_size: usize) -> Vec<String> {
        self.data_paths
            .choose_multiple(&mut rand::thread_rng(), batch_size)
            .map(|x| x.clone())
            .collect()
    }
}

pub struct DataLoader {
    pub dataset: Dataset,
    pub batch_size: usize,
}

impl DataLoader {
    pub fn new(ds: Dataset, bs: usize) -> DataLoader {
        DataLoader {
            dataset: ds,
            batch_size: bs,
        }
    }

    pub fn get_batch(&self) -> Array4F {
        let paths = self.dataset.batch_paths(self.batch_size);
        let mut vi: Array4F = Array4F::zeros([0, 0, 0, 0]);

        for path in paths {
            let reader = ImageReader::open(path).unwrap();
            let img = reader.decode().unwrap();
            let rgb = img.into_rgb32f();

            vi.push(Axis(0), rgb.into_ndarray3().view()).unwrap();
        }
        vi
    }
}
use image::io::Reader as ImageReader;
use image::RgbImage;
use rand::seq::SliceRandom;
use std::fs;
use std::path::Path;

pub enum InputType {
    Image,
    // Video,
    // Text,
    // Num,
    //
}

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

            input_type: InputType::Image,
        })
    }

    pub fn batch_paths(&self, batch_size: usize) -> Vec<String> {
        self.data_paths
            .choose_multiple(&mut rand::thread_rng(), batch_size)
            .map(|x| x.clone())
            .collect()
    }
}

struct DataLoader {
    dataset: Dataset,

    batch_size: usize,
    data_buff: Vec<RgbImage>,
}

impl DataLoader {
    fn new(ds: Dataset, bs: usize) -> DataLoader {
        DataLoader {
            dataset: ds,
            batch_size: bs,
            data_buff: Vec::new(),
        }
    }

    fn get_batch(&self) -> Vec<RgbImage> {
        let paths = self.dataset.batch_paths(self.batch_size);
        let mut vi = Vec::<RgbImage>::new();
        for path in paths {
            let reader = ImageReader::open(path).unwrap();
            let img = reader.decode().unwrap();
            let rgb = img.into_rgb8();

            vi.push(rgb);
        }
        vi
    }
}

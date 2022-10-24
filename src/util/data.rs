use image::{RgbImage};
use image::io::Reader as ImageReader;

use rand::seq::SliceRandom;
use std::fs;

pub enum InputType {
    Image,
    // Video,
    // Text,
    // Num,
    // 
}


pub enum DataLoadingError<'a>{
    EmptyDir(&'a str),
    UnexpSubdir()
}

pub struct Dataset {
    data_dir: String,
    data_paths: Vec<String>,

    input_type: InputType,
}


impl Dataset{
    fn new(data_dir: &str) -> Result<Dataset, DataLoadingError>{
        let mut pv: Vec<String>;

        let dir = fs::read_dir(data_dir).expect(format!("Could not load path: {}", data_dir).as_str());

        for item in dir{
            let pth = item.unwrap().path();
            
            if pth.is_file(){
                pv.push(pth.to_str().unwrap().to_string());
            }
        }

        if pv.is_empty(){
            return Err(DataLoadingError::<'b>::EmptyDir(format!("{} is an empty directory", data_dir).as_str()))
        }

        Ok(Dataset {
            data_dir: data_dir.to_string(),
            data_paths: pv,

            input_type: InputType::Image
        })
    }

    fn batch_paths(&self, batch_size: usize) -> Vec<String> {
        self.data_paths.choose_multiple(&mut rand::thread_rng(), batch_size).map(|x| x.clone()).collect()
    }


}




struct DataLoader{
    dataset: Dataset,

    batch_size: usize,
    data_buff: Vec<RgbImage>
}

impl DataLoader {
    fn get_batch(&self) -> Vec<RgbImage>{
        let paths = self.dataset.batch_paths(self.batch_size);
        let mut vi = Vec::<RgbImage>::new();
        for path in paths{
            let reader = ImageReader::open(path).unwrap();
            let img = reader.decode().unwrap();
            let rgb = img.into_rgb8();

            vi.push(rgb);
        }
        vi
    }
}
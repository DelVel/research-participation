data_dir=./dataset/coco2014
if [[ -d "$data_dir" ]]; then
  echo "Directory $data_dir exists. Skipping download."
  exit 0
fi
if ! command -v unzip &>/dev/null; then
  echo "unzip is not installed. Please install it and try again."
  exit 1
fi
echo "Directory $data_dir does not exist. Downloading..."
wget -P "$data_dir" -i coco2014_urls.txt
cd "$data_dir" || {
  echo "Failed to cd $data_dir"
  exit 1
}
unzip -d train2014 train2014.zip
unzip -d val2014 val2014.zip
unzip -d test2014 test2014.zip
unzip -d annotations_trainval2014 annotations_trainval2014.zip
unzip -d image_info_test2014 image_info_test2014.zip
rm ./*.zip
echo "Download complete."

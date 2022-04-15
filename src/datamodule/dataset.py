#  Copyright 2022 Taegyu Park
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

dataset_root = './dataset/coco2014'
train_val_annotations_root = f'{dataset_root}/annotations_trainval2014' \
                             f'/annotations'
train_caption = f'{train_val_annotations_root}/captions_train2014.json'
val_caption = f'{train_val_annotations_root}/captions_val2014.json'
test_info = f'{dataset_root}/image_info_test2014/annotations' \
            f'/image_info_test2014.json'
train_root = f'{dataset_root}/train2014'
val_root = f'{dataset_root}/val2014'
test_root = f'{dataset_root}/test2014'

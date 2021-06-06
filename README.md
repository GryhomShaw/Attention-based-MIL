- ![image](https://github.com/GryhomShaw/Attention-based-MIL/blob/master/Architecture.jpg)

- **Install:**

  - PyTorch=1.4.0
  - Install dependencies:` pip install -r requirements.txt`

- **Data preparation** 

  - A total of 800 TCT cell slices  (500 pos, 300 neg)

  - The data is in kfb format and needs to be read using the SDK designated:

    - `source kfbreader.rc `

  - **Sampling pathological slices:**

    - **Sample Pos:** 

      - `python sample_pos.py -o='your output path' -s=512`
      - **Args s**: size of each patch

    - **Sample Neg:** For class balance, sample the same number of samples from negative slices

      - `python sample_neg.py -o='your output path' -t=180 -s=512`
      - **Args t**: sampleing times for each slice

    - **Directory tree:** For both positive and negative samples, the same directory structure as the COCO dataset is used

      - ```
        demo_sample_pos
        ├── annotations
        │   ├── train.json
        │   └── valid.json
        └── images
            ├── T2019_108
            └── T2019_710
        ```

      - File naming rules:  sliceName_(x-y-width-height)\_label.jpg (e.g. <u>T2019_108\_(23054-32121-512-512)\_pos.jpg</u> )

  - **Bag generator:**  We treat the sample from the slice as a bag, thereby generating  instances for each bag

    - `python cut_img.py -o='../demo_bags' -s=128`

    - The path of the positive sample and the negative sample needs to be specified in the file

    - train_valid_split: For convenience, the data set can be divided directly (refer to **utils/train_vaild_split.py**)

    - **Directory tree:**

      - ```
        demo_bags
        ├── neg
        │   ├── T2019_397_(10301-1345-10813-1857)
        │   ├── ...
        │   └── T2019_408_(9962-5703-10474-6215)
        ├── pos
        │   ├── T2019_108_(10074-30215-512-512)
        │   ├── ...
        │   └── T2019_710_(9809-22595-512-512)
        ├── train_vaild_split.json
        └── train_vaild_split.py
        ```

- **Train：**

  - **Single GPU:**
    - You can directly run the provided script for training: `./scripts/run_demo.sh`
    - **Args m：** type of encoder (including Mobilenet and Resnet series models)
    - It is worth noting that ,if you need to add instance-level constraints to the network, you need to specify the **instance_eval_bilateral** or **instance_eval_unilateral parameters** (two different constraint methods, and specify the number of samples **(k_sample**)
  - **Model parallel training:**
    - We uses a multi-GPU model parallel method to solve the problem of insufficient memory for end-to-end training of the model, and uses a pipeline method to accelerate model training and improve model training efficiency.
    - You can directly run the provided script for training: `./scripts/run_demo_mp.sh`
    - **Args split_index_list:** The parameter **split_index_list** is used to determine whether to use model parallel training. If it is None, it means not to use; otherwise, it means the index used by the encoder for splitting.The unit of network splitting is bottleneck, see the corresponding file for details (**modes/backbone/mobilenetv2.py** or **modes/backbone/resnet.py** )

- **Inference：**

  -   `./scripts/run_demo.sh`

  - **Directory tree:** For each positive sample, the result is composed of the original image, heatmap and ground truth.

    ```
    test_output
    └── demo
        ├── neg
        │   ├── T2019_397_(10301-1345-10813-1857)_neg_color.jpg
        │   └── T2019_397_(10301-1345-10813-1857)_neg.jpg
        └── pos
            ├── T2019_710_(9553-21827-512-512)_pos_color.jpg
            ├── T2019_710_(9553-21827-512-512)_pos.jpg
            └── T2019_710_(9553-21827-512-512)_pos_labeled.jpg
    ```

  - The root directory of positive and negative samples and ground truth need to be specified in the file(**inference_entire.py**)

  - You can use the **utils/viaual.py**  to visualize the annotation results：

    `python visual.py -i='../demo_sample_pos' -o='./demo_labeled'`
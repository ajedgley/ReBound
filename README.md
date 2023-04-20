# ReBound
## An Open-Source LiDAR Visualization and Annotation Tool

ReBound aims to reduce the time AV researchers spend dealing with dataset API issues so they can spend more time analyzing data. Researchers are able to load in data and immediately visualize it as LiDAR point clouds or RGB images.

Our app aims to provide researchers with the necessary tools to easily visualize the data their models are producing so that they can quickly find and see faults with their own eyes to get a better idea of why their model is failing.



### Features

- Interactive GUI based scene navigation

- Real time sensor switching

- Automated fault scanning (False Positives, Incorrect Annotations, Unmatched GT Annotations)

- Tested for Windows and Linux

- Real time annotation filtering by class, confidence score, or error type

- Export rendered figures to PNG

- Modular system to support expansion to multiple dataset formats



### Docs: Documentation for ReBound can be found in the Wiki [here](https://github.com/ajedgley/ReBound/wiki/ReBound-Intro)



<img title="" src="https://files.gitbook.com/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FLwXJQFLIM5IJpwSLEYfC%2Fuploads%2F9YIBroiQ3eduUX55VuD7%2FLVT1.png?alt=media&token=d06c7735-5c6a-4ba6-9ffd-79e266c5fdf0" alt="" data-align="center">

If you find this codebase useful, please consider citing:

    @article{chen2023rebound,
      title={ReBound: An Open-Source 3D Bounding Box Annotation Tool for Active Learning},
      author={Chen, Wesley and Edgley, Andrew and Hota, Raunak and Liu, Joshua and Schwartz, Ezra and Yizar, Aminah and Peri, Neehar and Purtilo, James},
      journal={AutomationXP @ CHI 2023},
      year={2023},
    }

# Data Project Part-1
We will learn about how to structure the data projects, for productionizing your Data Science &amp; ML work.
This is the basic folder structure without any dependency manager, config files, dvc, pdoc etc. We will slowly introduce each as the need arises.


    .
    ├── data                    # data files location
    │   ├── final               # Store final clean and grouped data here.
    │   ├── processed           # Store processed files here, intermediate files.
    │   └── raw                 # Store raw data files here.
    ├── docs                    # Store your project related documents here, such as project report, ideas
    ├── models                  # Store the models here
    ├── notebooks               # Store python notebooks here
    ├── src                     # Source files
    ├── tests                   # Automated tests (alternatively `spec` or `tests`)
    └── README.md               # A instructions file.

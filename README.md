# Towards production-ready end-to-end Federated Learning for automotive applications  


Author | Swetha Lakshmana Murthy
--- | --- |
Project Type | Master Thesis 
Course Name ; University | M.Sc Infotech, Universität Stuttgart
Supervisor | Dr. Denis Kramer
Start Date | 16th May, 2022
End Data | 14th November, 2022


## Introduction 
An implementation of FL for the two automotive use cases,

**Use-case (a)** - Actual battery capacity prediction for EVs  
**Use-case (b)** - Energy-demand forecasting for CSs  

## Code Structure
1. The code for the use-case (a) is present in the EV folder. It further contains two sub-folders, one without end-to-end DP and the other with the differential privacy model.  
2. The code for the ise-case (b) is present in the CS folder.  
3. Sample SUMO simulation files are present in the SUMO_Simulator folder.  

Please refer the thesis report for a detailed understanding of the system set-up.

## Steps to run the code
Please install the packages as in the requirements.txt file.  
Please navigate to the respective use-case folder and adapt the path for accessing and storing the data accordingly. Further the main.py file needs to be executed to start the FL process.  

```
pip install -r requirements.txt
```

```
python main.py --iteration enter_number_of_iterations
```
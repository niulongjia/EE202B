# EE202B

### Project Description
In our project, we discuss and quantitatively evaluate the capability of digital electricity meters to be used for human occupancy/human activities prediction in various models based on ECO Data-set house#2. Given the fact that digital meters are widely installed and does not impose additional costs on the residents, there are huge opportunities to implement it on real automation systems.

### ECO Dataset
ECO ( Electricity Consumption and Occupancy) data set is the first data set available that contains both electricity consumption and ground truth occupancy information of households.   
The data collection and analysis are done by the team of prof. Wilhelm Kleiminger, their work is available through:   
https://www.vs.inf.ethz.ch/res/show.html?what=eco-data   
***The data set is available through:***   
***http://data-archive.ethz.ch/delivery/DeliveryManagerServlet?dps_pid=IE594964***  

### Dataset Structure
I have downloaded the dataset and it is placed under Eco_Dataset/house#(number 1 ~ 5). Note that I have changed the dataset structure, so if you want to download it yourself, you must follow exactly same structure as I do.   

### Course Project Website
***our website is available at:   
https://sites.google.com/view/activities-prediction-202b***   

**The data visualization graphs are generated with the following python scripts under Eco_Dataset/codes:**
***graph_house1_plugs.py   
graph_house1_sm.py   
graph_house2_plugs.py   
graph_house2_sm.py   
graph_house3_plugs.py   
graph_house3_sm.py   
graph_house4_plugs.py   
graph_house4_sm.py   
graph_house5_plugs.py   
graph_house5_sm.py***

The graphs are generated with the following python scripts under Eco_Dataset/codes:   
***house2_plug_features_occupancy.py*** (use plugs power information to predict occupancy)   
***house2_sm_features_occupancy.py*** (use smart meter power information to predict occupancy)   
***house2_sm_features_activities.py*** (use smart meter power information to predict activities)   
***house2_sm_features_activities_new.py*** (use smart meter power distribution information to predict activities)   
***house2_clf_predict_house1_occupancy.py*** (use model trained with house#02 smart meter to predict other house#01)   
***house2_clf_predict_house3_occupancy.py*** (use model trained with house#02 smart meter to predict other house#03)   
***house2_clf_predict_house4_occupancy.py*** (use model trained with house#02 smart meter to predict other house#04)   
***house2_clf_predict_house5_occupancy.py*** (use model trained with house#02 smart meter to predict other house#05)   


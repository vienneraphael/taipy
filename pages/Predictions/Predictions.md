<|layout|columns=2 9|gap=50px|
<sidebar|sidebar|
Create and select **scenarios**

<|{selected_scenario}|scenario_selector|>
|sidebar>

<scenario|part|render={selected_scenario}|
# **Prediction**{: .color-primary} page

<<<<<<< HEAD
Predict **company sales** depending on the holidays that employees take. Create different scenarios and choose the best one.

Here are two dataframes representing employee holidays ([High](data/holiday_high.csv) and [Low](data/holiday_low.csv)) that you can upload to the application.

<|3 5|layout|
<date|
#### Level

A parameter to choose how holidays impact your predictions.
=======
<|1 1|layout|
<date|
#### Level

A parameter to choose how pessimistic or optimistic your predictions will be.
>>>>>>> 30fd282 (proposition de changement)

<|{selected_level}|slider|on_change=on_change_params|not continuous|min=70|max=150|>
|date>

<country|
#### **Holiday**{: .color-primary}

<<<<<<< HEAD
Upload the CSV of employee holidays:

<|{dn_holiday}|data_node|expanded=False|>


<|{selected_holiday}|file_selector|label=Holiday|on_action=on_change_params|>
=======
Choose if there is an holiday coming

<|{selected_holiday}|toggle|label=Holiday|on_change=on_change_params|>
>>>>>>> 30fd282 (proposition de changement)
|country>
|>

Run your scenario

<|{selected_scenario}|scenario|on_submission_change=on_submission_change|not expanded|>

---------------------------------------

<<<<<<< HEAD
## **Predictions**{: .color-primary}

<|{dn_result}|data_node|>
=======
## **Predictions**{: .color-primary} and explorer of data nodes

<|Data Nodes|expandable|
<|1 5|layout|
<|{selected_data_node}|data_node_selector|> 

<|{selected_data_node}|data_node|>
|>
|>

>>>>>>> 30fd282 (proposition de changement)
|scenario>
|>

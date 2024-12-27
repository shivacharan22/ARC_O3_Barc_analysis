import streamlit as st
import matplotlib.pyplot as plt
import arckit
import json
import pickle
from dataclasses import dataclass

@dataclass
class ARCData:
    problem_keys: list
    colors: dict

def update_ids(Problem_keys_o3_got_Wrong, Barc_attempt):
    """
    Updates a list of task IDs with their corresponding output grids.
    
    Args:
        Problem_keys_o3_got_Wrong (list): List of task IDs that were incorrectly solved
        Barc_attempt (list): List of output grids from the Barc model's attempts
        
    Returns:
        list: List of tuples containing (task_id, output_grid) pairs
    """
    up_list = []
    for id, out_grid in zip(Problem_keys_o3_got_Wrong, Barc_attempt):
        up_list.append((id, out_grid))
    return up_list

def correct_incorrect_tasks(tasks, attemps):
    """
    Separates tasks into correct and incorrect lists based on a specific task ID.
    
    Args:
        Problem_keys_o3_got_Wrong: List of task IDs to process
        
    Returns:
        tuple: (data_correct, data_wrong) where:
            - data_correct: List containing the task data for ID 'a3f84088' if present
            - data_wrong: List containing task data for all other IDs
    """
    data_correct = []
    data_wrong = []
    for id,Barc_grid in tasks:
        o3_output = attemps[id][0]['attempt_1']
        if id  == "a3f84088":
            task = arckit.load_single(id)
            task_dict = task.to_dict()
            task_dict['test'][0]['o3_output'] = o3_output
            task_dict['test'][0]['Barc_output'] = Barc_grid
            data_correct.append(task_dict)
        else:
            task = arckit.load_single(id)
            task_dict = task.to_dict()
            task_dict['test'][0]['o3_output'] = o3_output
            task_dict['test'][0]['Barc_output'] = Barc_grid
            data_wrong.append(task_dict)
    return data_correct, data_wrong

def plot_circle_grid(grid, title, arc_data, ax=None):
    """Visualize a grid with circles on a gray background, including matplotlib figure background."""
    if ax is None:
        #fig, ax = plt.subplots(figsize=(4, 4))  # Adjusting figure size for smaller grids
        fig, ax = plt.subplots(figsize=(max(4, len(grid[0]) // 2), max(4, len(grid) // 2)))
        fig.patch.set_facecolor("#4a4a4a")  # Set the background color of the entire figure
    
    ax.set_facecolor("#4a4a4a")  # Set a medium gray background color for the grid area

    for i, row in enumerate(grid):
        for j, value in enumerate(row):
            color = arc_data.colors.get(value, "black")  # Circle color based on value
            circle = plt.Circle((j, -i), 0.4, color=color)  # Slightly larger circles for clarity
            ax.add_artist(circle)

    # Set axis limits and aspect ratio
    ax.set_xlim(-1, len(grid[0]))
    ax.set_ylim(-len(grid), 1)
    ax.axis("off")  # Turn off the axis
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, color="white", fontsize=14, fontweight="bold")  # Title text color and style

def display_train_data(task, arc_data):
    """Display train inputs and outputs in columns."""
    st.subheader("Train Data")
    train_data = task["train"]

    # Create a column for each train example
    columns = st.columns(len(train_data))

    for col, example in zip(columns, train_data):
        input_grid = example["input"]
        output_grid = example["output"]

        with col:
            st.write("Input")
            fig, ax = plt.subplots(figsize=(4, 4))
            plot_circle_grid(input_grid, "", arc_data, ax=ax)
            st.pyplot(fig)

            st.write("Output")
            fig, ax = plt.subplots(figsize=(4, 4))
            plot_circle_grid(output_grid, "", arc_data, ax=ax)
            st.pyplot(fig)

def display_test_data(task, arc_data):
    """Display test data in columns."""
    st.subheader("Test Data")

    # Extract test grids
    test_input = task["test"][0]["input"]
    o3_output = task["test"][0]["o3_output"]
    Barc_output = task["test"][0]["Barc_output"]
    ground_truth = task["test"][0]["output"]

    # Create columns for test data
    columns = st.columns(4)

    # Test Input
    with columns[0]:
        st.write("Test Input")
        fig, ax = plt.subplots(figsize=(4, 4))
        plot_circle_grid(test_input, "", arc_data, ax=ax)
        st.pyplot(fig)

    # O3 Model Output
    with columns[1]:
        st.write("O3 Output")
        fig, ax = plt.subplots(figsize=(4, 4))
        plot_circle_grid(o3_output, "", arc_data, ax=ax)
        st.pyplot(fig)

    # Barc Model Output
    with columns[2]:
        st.write("Barc Output")
        fig, ax = plt.subplots(figsize=(4, 4))
        plot_circle_grid(Barc_output, "", arc_data, ax=ax)
        st.pyplot(fig)

    # Ground Truth
    with columns[3]:
        st.write("Ground Truth")
        fig, ax = plt.subplots(figsize=(4, 4))
        plot_circle_grid(ground_truth, "", arc_data, ax=ax)
        st.pyplot(fig)

def main():
    st.set_page_config(layout="wide")
    st.title("ARC Task Visualization App")

    # Load data
    with open('data/attemps.json') as f:
        o3_outputs = json.load(f)
    with open('data/data_Barc_answers.pkl', 'rb') as f:
        Barc_attempt = pickle.load(f)

    # Initialize the ARCData instance
    arc_data = ARCData(
        problem_keys=[
            "da515329", "f9d67f8b", "891232d6", "52fd389e", "c6e1b8da", 
            "09c534e7", "ac0c5833", "47996f11", "b457fec5", "b7999b51", 
            "b9630600", "896d5239", "40f6cd08", "8b28cd80", "93c31fbe", 
            "25094a63", "05a7bcf2", "0934a4d8", "79fb03f4", "4b6b68e5", 
            "aa4ec2a5", "1acc24af", "f3b10344", "256b0a75", "d931c21c", 
            "16b78196", "a3f84088", "212895b5", "0d87d2a6", "3ed85e70", 
            "e619ca6e", "e1d2900e", "d94c3b52", "e681b708"
        ],
        colors={
            0: "black",  # background
            1: "blue",
            2: "red",
            3: "green",
            4: "yellow",
            5: "gray",
            6: "magenta",
            7: "orange",
            8: "skyblue",
            9: "brown",
        }
    )
    # Process tasks
    tasks = update_ids(arc_data.problem_keys, Barc_attempt)
    correct_tasks, wrong_tasks = correct_incorrect_tasks(tasks, o3_outputs)

    # Tabs for category selection
    tabs = st.tabs(["Correct Tasks", "Incorrect Tasks"])

    # Correct Tasks Tab
    with tabs[0]:
        st.header("Correct Tasks (O3 Model Failed, Barc Model Correct)")
        task_options = [task['id'] for task in correct_tasks]
        selected_task_id = st.selectbox(
            "Choose a task to view:",
            options=task_options,
            format_func=lambda x: f"Task {x}"
        )
        if selected_task_id:
            selected_task = next(task for task in correct_tasks if task["id"] == selected_task_id)
            st.subheader(f"Selected Task: {selected_task_id}")
            display_train_data(selected_task, arc_data)
            display_test_data(selected_task, arc_data)

    # Incorrect Tasks Tab
    with tabs[1]:
        st.header("Incorrect Tasks (Both Models Failed)")
        task_options = [task['id'] for task in wrong_tasks]
        selected_task_id = st.selectbox(
            "Choose a task to view:",
            options=task_options,
            format_func=lambda x: f"Task {x}"
        )
        if selected_task_id:
            selected_task = next(task for task in wrong_tasks if task["id"] == selected_task_id)
            st.subheader(f"Selected Task: {selected_task_id}")
            display_train_data(selected_task, arc_data)
            display_test_data(selected_task, arc_data)

if __name__ == "__main__":
    main()
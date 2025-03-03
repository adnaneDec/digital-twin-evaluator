import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import requests
from streamlit_gsheets import GSheetsConnection
import uuid
import openai

# Establish a Google Sheets connection
conn = GSheetsConnection(connection_name="gsheets")  # Match the name in secrets.toml

# Fetch existing data from all worksheets
profile_df = conn.read(worksheet="profile_data", usecols=list(range(9)), ttl=500)  # Adjust `range` to match your columns
profile_df = profile_df.dropna(how="all")

scores_df = conn.read(worksheet="scores", usecols=list(range(6)), ttl=500)  # Adjust `range` as per column count
scores_df = scores_df.dropna(how="all")

comments_df = conn.read(worksheet="comments", usecols=list(range(5)), ttl=500)  # Adjust `range` as per column count
comments_df = comments_df.dropna(how="all")

# Generate unique session ID if not already present
if "unique_id" not in st.session_state:
    st.session_state["unique_id"] = str(uuid.uuid4())

# Sidebar navigation with logos
st.sidebar.image(
    "https://cdn.brandfetch.io/idNkQH2wyM/w/401/h/84/theme/dark/logo.png?c=1bfwsmEH20zzEfSNTed",
    use_container_width=True,
)

st.sidebar.image(
    "https://cdn.brandfetch.io/id79psAFnq/theme/dark/logo.svg?c=1bfwsmEH20zzEfSNTed",
    use_container_width=True,
)

def profile_identification():
    # Ensure profile_data exists in session state
    if "profile_data" not in st.session_state:
        st.session_state["profile_data"] = {
            "field_of_work": "Research",
            "years_experience": 0,
            "current_system": "",
            "position": "",
            "country": "",
            "department": "",
            "comments": ""
        }

    # Populate fields with existing session state values
    st.write("Before we dive in, let us know each other a little bit 🙂. The results of this questionnaire are to be collected for research purposes.")
    st.session_state["profile_data"]["field_of_work"] = st.radio(
        "What is your field of work?", 
        ["Research", "Industry"], 
        index=["Research", "Industry"].index(st.session_state["profile_data"].get("field_of_work", "Research"))
    )
    st.session_state["profile_data"]["years_experience"] = st.slider(
        "How many years have you worked on Information Systems (IS) or Warehouse Management Systems (WMS)?", 
        0, 50, step=1, 
        value=st.session_state["profile_data"].get("years_experience", 0)
    )
    st.session_state["profile_data"]["current_system"] = st.text_input(
        "What is the name of the IS or WMS you use (if any)?", 
        value=st.session_state["profile_data"].get("current_system", "")
    )
    st.session_state["profile_data"]["position"] = st.text_input(
        "What is your current position?", 
        value=st.session_state["profile_data"].get("position", "")
    )
    st.session_state["profile_data"]["country"] = st.text_input(
        "Which country are you based in?", 
        value=st.session_state["profile_data"].get("country", "")
    )
    st.session_state["profile_data"]["department"] = st.text_input(
        "What department are you affiliated with (e.g., R&D, Logistics)?", 
        value=st.session_state["profile_data"].get("department", "")
    )
    st.session_state["profile_data"]["comments"] = st.text_area(
        "Any additional comments or insights you would like to share?", 
        value=st.session_state["profile_data"].get("comments", "")
    )

    return st.session_state["profile_data"]

# Radar Chart Function using Plotly
def plot_radar_chart(data, categories):
    fig = go.Figure()

    # Add radar trace
    fig.add_trace(go.Scatterpolar(
        r=data,
        theta=categories,
        fill='toself',
        name='Average Scores',
        line=dict(color='royalblue', width=3),
        fillcolor='rgba(65, 105, 225, 0.4)'
    ))

    # Update layout for aesthetics
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 5],
                tickvals=[0, 1, 2, 3, 4, 5],
                tickfont=dict(size=12),
                gridcolor='lightgray'
            ),
            angularaxis=dict(
                tickfont=dict(size=12),
                gridcolor='lightgray'
            ),
        ),
        showlegend=False,
        template='plotly_white',
        title=dict(
            text='Average Scores by Category',
            x=0.5,
            font=dict(size=18, color='royalblue')
        )
    )

    # Display the radar chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)

# Define the evaluation framework with fuzzy categories
evaluation_framework = {
    "Core Digital Twin Characteristics": {
        "description": "Digital Twins (DTs) are virtual representations of physical entities, enabling seamless bi-directional data exchange for real-time monitoring, simulation, and decision-making. They allow for lifecycle management of systems, enhancing their operational, predictive, and resilience capabilities.",
        "subcategories": [
            {
                "subcategory": "Physical Entity",
                "questions": [
                    {"question": "How clearly defined is the real-world physical object/system?", "type": "fuzzy"},
                    {"question": "How well are the entity’s boundaries and hierarchical levels specified?", "type": "fuzzy"},
                    {"question": "How granular is the representation in capturing detailed interactions?", "type": "fuzzy"}
                ]
            },
            {
                "subcategory": "Virtual Entity",
                "questions": [
                    {"question": "How accurate is the virtual counterpart of the physical entity?", "type": "fuzzy"},
                    {"question": "How comprehensive is the virtual representation (geometry, behavior, rules)?", "type": "fuzzy"},
                    {"question": "How real-time is the update of the virtual entity?", "type": "fuzzy"}
                ]
            }
        ]
    },
    "Connectivity and Synchronization": {
        "description": "A fundamental feature of DTs is their ability to maintain dynamic, bi-directional connections between physical and virtual entities. This involves ensuring synchronization through real-time or near-real-time data flows to support operational and strategic objectives.",
        "subcategories": [
            {
                "subcategory": "Physical-to-Virtual Connection",
                "questions": [
                    {"question": "How seamless is the transfer of real-time sensor data to the virtual model?", "type": "fuzzy"},
                    {"question": "How assured is the quality of the transferred data?", "type": "fuzzy"}
                ]
            },
            {
                "subcategory": "Virtual-to-Physical Feedback",
                "questions": [
                    {"question": "How effective is the feedback mechanism from the virtual to physical entity?", "type": "fuzzy"},
                    {"question": "How real-time is the decision-making feedback to optimize operations?", "type": "fuzzy"}
                ]
            }
        ]
    },
    "Simulation and Decision Support": {
        "description": "Digital Twins enable predictive and prescriptive capabilities through simulations, providing actionable insights and optimization strategies. These capabilities align with transitioning decision-making from reactive to proactive processes.",
        "subcategories": [
            {
                "subcategory": "Simulation Capabilities",
                "questions": [
                    {"question": "How well can the system evaluate 'what-if' scenarios?", "type": "fuzzy"},
                    {"question": "How effective is the system in predictive and prescriptive analysis?", "type": "fuzzy"}
                ]
            },
            {
                "subcategory": "Decision-Making and Autonomy",
                "questions": [
                    {"question": "How actionable are the system's insights for decision-making?", "type": "fuzzy"},
                    {"question": "How autonomous is the system in executing decisions?", "type": "fuzzy"}
                ]
            }
        ]
    },
    "Data Management and Integration": {
        "description": "DTs depend on robust data collection, integration, and processing frameworks to ensure seamless real-time operations. This includes IoT devices, cloud/edge computing, and compatibility with enterprise systems.",
        "subcategories": [
            {
                "subcategory": "Data Collection and Processing",
                "questions": [
                    {"question": "How reliable is the collection of continuous, real-time data?", "type": "fuzzy"},
                    {"question": "How effective is the fusion of data from multiple sources?", "type": "fuzzy"}
                ]
            },
            {
                "subcategory": "System Integration",
                "questions": [
                    {"question": "How well is the DT integrated with enterprise systems?", "type": "fuzzy"},
                    {"question": "How interoperable is the DT with other systems?", "type": "fuzzy"}
                ]
            }
        ]
    },
    "Learning and Adaptability": {
        "description": "A mature DT leverages AI and machine learning to self-improve, recognize context changes, and adapt its models autonomously. This adaptability ensures scalability and relevance throughout its lifecycle.",
        "subcategories": [
            {
                "subcategory": "Context Awareness",
                "questions": [
                    {"question": "How dynamically does the system recognize environmental changes?", "type": "fuzzy"},
                    {"question": "How well are uncertainties and interactions incorporated?", "type": "fuzzy"}
                ]
            },
            {
                "subcategory": "Learning Capabilities",
                "questions": [
                    {"question": "How effectively does the system use AI/ML for optimization?", "type": "fuzzy"},
                    {"question": "How autonomously does the system self-diagnose and update models?", "type": "fuzzy"}
                ]
            }
        ]
    },
    "Fidelity and Validation": {
        "description": "DTs aim for high-fidelity representations while balancing computational efficiency. Validation ensures their trustworthiness and alignment with physical behaviors, critical for stakeholder confidence.",
        "subcategories": [
            {
                "subcategory": "Model Fidelity",
                "questions": [
                    {"question": "How appropriate is the fidelity level for the DT’s goals?", "type": "fuzzy"},
                    {"question": "How effectively are computational constraints managed?", "type": "fuzzy"}
                ]
            },
            {
                "subcategory": "Verification and Feedback",
                "questions": [
                    {"question": "How rigorously has the virtual model been validated?", "type": "fuzzy"},
                    {"question": "How effectively are real-world outcomes used for model refinement?", "type": "fuzzy"}
                ]
            }
        ]
    },
    "Digital Twin Services": {
        "description": "The functional utility of DTs is measured by their service capabilities, such as real-time monitoring, predictive maintenance, and operational optimization, aimed at enhancing the system's performance and resilience.",
        "subcategories": [
            {
                "subcategory": "Monitoring and Real-Time Feedback",
                "questions": [
                    {"question": "How well does the system provide real-time monitoring?", "type": "fuzzy"},
                    {"question": "How effectively does the system detect anomalies and propose corrective actions?", "type": "fuzzy"}
                ]
            },
            {
                "subcategory": "Optimization and Control",
                "questions": [
                    {"question": "How well does the system enable operational optimization?", "type": "fuzzy"},
                    {"question": "How effectively does the system support predictive maintenance?", "type": "fuzzy"}
                ]
            }
        ]
    },
    "Technological Readiness": {
        "description": "The deployment of DTs depends on the integration of advanced technologies such as IoT, cloud computing, AI/ML, and ensuring cybersecurity. Scalability and compliance with privacy standards are also critical considerations.",
        "subcategories": [
            {
                "subcategory": "Enabling Technologies",
                "questions": [
                    {"question": "How effectively are advanced technologies (IoT, AI/ML) employed?", "type": "fuzzy"},
                    {"question": "How well does the system incorporate new developments like blockchain?", "type": "fuzzy"}
                ]
            },
            {
                "subcategory": "Security and Privacy",
                "questions": [
                    {"question": "How effectively are cybersecurity risks addressed?", "type": "fuzzy"},
                    {"question": "How scalable is the system to handle data volumes and functionalities?", "type": "fuzzy"}
                ]
            }
        ]
    }
}

# Initialize session state
if "scores" not in st.session_state:
    st.session_state.scores = {
        category: {item["subcategory"]: [None] * len(item["questions"]) for item in data["subcategories"]}
        for category, data in evaluation_framework.items()
    }
if "comments" not in st.session_state:
    st.session_state.comments = {
        category: {item["subcategory"]: "" for item in data["subcategories"]}
        for category, data in evaluation_framework.items()
    }
if "profile_data" not in st.session_state:
    st.session_state.profile_data = {}


# Validation function
def validate_all_answers():
    """Ensure all questions are answered by checking for None values."""
    for category, subcategories in st.session_state.scores.items():
        for subcat, answers in subcategories.items():
            if None in answers:  # Check for uninitialized (None) values
                return False, f"Please complete all questions in '{category}' - '{subcat}'."
    return True, ""

# Sidebar navigation
st.sidebar.title("Digital Twin Evaluation Framework")
page = st.sidebar.radio(
    "Select a Category",
    ["Profile Identification"] + list(evaluation_framework.keys()) + ["Summary"],
    key="navigation"
)

# Update session state when navigating via radio
st.session_state["current_page"] = page

# Scroll to top when changing pages
if "last_page" not in st.session_state:
    st.session_state["last_page"] = page

if st.session_state["last_page"] != page:
    st.session_state["last_page"] = page
    st.markdown(
        """
        <script>
        window.scrollTo(0, 0);
        </script>
        """,
        unsafe_allow_html=True
    )

# Main content rendering
st.title("Digital Twin Evaluation Framework")

if page == "Profile Identification":
    profile_data = profile_identification()
    st.session_state["profile_data"] = profile_data

elif page != "Summary":
    st.subheader(f"Category: {page}")
    st.write(evaluation_framework[page]["description"])

    for subcategory in evaluation_framework[page]["subcategories"]:
        st.write(f"### {subcategory['subcategory']}")
        for i, question in enumerate(subcategory["questions"]):
            if st.session_state.scores[page][subcategory["subcategory"]][i] is None:
                st.session_state.scores[page][subcategory["subcategory"]][i] = 0
            st.session_state.scores[page][subcategory["subcategory"]][i] = st.slider(
                question["question"],  # Extract the question text
                0,
                5,
                step=1,
                key=f"{page}_{subcategory['subcategory']}_{i}",
                value=st.session_state.scores[page][subcategory["subcategory"]][i]
            )
        comment = st.text_area(
            f"Comments for {subcategory['subcategory']}",
            key=f"comment_{page}_{subcategory['subcategory']}",
            value=st.session_state.comments[page][subcategory["subcategory"]]
        )
        st.session_state.comments[page][subcategory["subcategory"]] = comment

elif page == "Summary":
    st.subheader("Evaluation Summary")

    # Initialize summary data list
    summary_data = []

    # Calculate average scores for each subcategory
    for category, subcategories in st.session_state.scores.items():
        for subcat, answers in subcategories.items():
            avg_score = np.mean([a for a in answers if a is not None]) if answers else 0
            summary_data.append({
                "Category": category,
                "Subcategory": subcat,
                "Average Score": avg_score
            })

    # Convert summary data to a DataFrame and display as a table
    summary_df = pd.DataFrame(summary_data)
    st.table(summary_df)

    # Check if all questions have been answered
    valid, error_message = validate_all_answers()

    if not valid:
        st.error(error_message)
    else:
        st.write("""
            ### Ready to Proceed?
            The table above summarizes the evaluation scores for each category and subcategory.
            A radar graph below visualizes the average scores by category.
            Please click the 'Submit and Continue' button to share your results for further research. 😊
        """)

        # Prepare data for radar chart
        radar_data = summary_df.groupby("Category")["Average Score"].mean().tolist()
        radar_categories = summary_df["Category"].unique()

        # Plot radar chart
        plot_radar_chart(radar_data, radar_categories)

        if st.button("Submit and Continue"):
            # Send data to Google Apps Script Web App
            timestamp = pd.Timestamp.now().isoformat()
            
            # Prepare profile_data with unique_id
            profile_data = {**st.session_state["profile_data"], "timestamp": timestamp, "unique_id": st.session_state["unique_id"]}

            # Prepare scores
            scores_data = []
            for category, subcategories in st.session_state["scores"].items():
                for subcat, answers in subcategories.items():
                    for idx, score in enumerate(answers):
                        subcategory_data = next(
                            (sub for sub in evaluation_framework[category]["subcategories"] if sub["subcategory"] == subcat),
                            None
                        )
                        if subcategory_data:
                            question_text = subcategory_data["questions"][idx]["question"]
                        else:
                            raise ValueError(f"Subcategory '{subcat}' not found in category '{category}'.")
            
                        scores_data.append({
                            "timestamp": timestamp,
                            "unique_id": st.session_state["unique_id"],
                            "category": category,
                            "subcategory": subcat,
                            "question": question_text,
                            "score": score
                        })

            # Prepare comments
            comments_data = []
            for category, subcategories in st.session_state["comments"].items():
                for subcat, comment in subcategories.items():
                    comments_data.append({
                        "timestamp": timestamp,
                        "unique_id": st.session_state["unique_id"],
                        "category": category,
                        "subcategory": subcat,
                        "comment": comment
                    })
            
            # Send data to Google Sheets
            data_to_send = {
                "profile_data": profile_df,
                "scores": scores_df,
                "comments": comments_df,
                }
            
            # Convert dictionaries to a DataFrames
            profile_data_df = pd.DataFrame([profile_data])
            scores_data_df = pd.DataFrame(scores_data)
            comments_data_df = pd.DataFrame(comments_data)

            # Add the new data in the read datasets 
            updated_profiles = pd.concat([profile_df, profile_data_df], ignore_index=True)
            updated_scores = pd.concat([scores_df, scores_data_df], ignore_index=True)
            updated_comments = pd.concat([comments_df, comments_data_df], ignore_index=True)

            # Update Google Sheets with the new vendor data
            conn.update(worksheet="profile_data", data=updated_profiles)
            conn.update(worksheet="scores", data=updated_scores)
            conn.update(worksheet="comments", data=updated_comments)

            st.success("Your feedback has been successfully submitted!")
            
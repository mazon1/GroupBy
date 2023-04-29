import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
from sklearn.model_selection import train_test_split
from causalml.inference.tree import UpliftTreeClassifier
from causalml.metrics import plot_gain, auuc_score, plot_uplift_by_percentile

@st.cache
def load_data():
    df = pd.read_csv('dat/feature_eng_data.csv')
    return df

def welcome_page():
    st.title("Welcome to the Uplift Model Platform")
    st.write("""
    The Uplift Model Platform is designed to help businesses and individuals evaluate the effectiveness of their marketing campaigns, promotions, and customer engagement strategies. With our cutting-edge uplift modeling techniques, you can optimize your marketing efforts by identifying the most suitable target audience and maximizing return on investment.
    
    In this platform, you'll be able to:
    * Explore your campaign data
    * Evaluate the Uplift Model's Performance
    * Visualize and analyze the Model and Segmentation results
    * Identify your Persudables, Sure Things, Lost Causes and Sleeping dogs
    * Make data-driven decisions to improve your marketing strategy
    
    Let's get started! Choose an option from the radio button menu on the left to navigate through the platform.
    """)  

def create_uplift_tree_model(X_train, y_train, treatment_train, max_depth, min_samples_leaf):
    uplift_model = UpliftTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, 
                                        evaluationFunction='KL', control_name='control')
    uplift_model.fit(X_train, treatment_train, y_train)
    return uplift_model

def numerical_analysis():
    st.write("Numerical Analysis")

def categorical_analysis():
    st.write("Categorical Analysis")

def campaign_results():
    df = load_data()
    plot_data_df = df[["treatment_group_key", "conversion", "total_spend", "exposure"]].copy()
    plot_data_df["campaign"] = plot_data_df["treatment_group_key"].apply(lambda x: x.split("_")[0])
    plot_data_df["channel"] = plot_data_df["treatment_group_key"].apply(lambda x: x.split("_")[1])
    X = df.drop(columns=["conversion"])
    y = df[["conversion"]]
    treatment = df[["exposure"]]
    X_train, X_test, y_train, y_test, trmnt_train, trmnt_test = train_test_split(X, y, treatment, test_size=0.2, random_state=42)
    uplift_model = create_uplift_tree_model(X_train, y_train, trmnt_train, max_depth=5, min_samples_leaf=100)
    plot_data_df["uplift"] = uplift_model.predict(X)[:, 1] - uplift_model.predict(X)[:, 0]
    plot_data_df["class"] = plot_data_df["uplift"].apply(lambda x: "Persuadable" if x > 0 else "Sure Thing" if x == 0 else "Lost Cause" if x < 0 and x > -0.1 else "Sleeping Dog")
    plot_data_df["class"] = pd.Categorical(plot_data_df["class"], categories=["Persuadable", "Sure Thing", "Lost Cause", "Sleeping Dog"])
    plot_data_df.sort_values("uplift", inplace=True)
    return df, plot_data_df, X_test, y_test, trmnt_test

def create_uplift_chart(plot_data_df):
    """
    Creates an uplift chart using the plot_data_df dataframe.

    Parameters:
    plot_data_df (pandas.DataFrame): Dataframe containing columns `group`, `n_treatment`, `n_control`, and `uplift`.

    Returns:
    altair.vegalite.v4.api.Chart: An Altair chart displaying the uplift chart.
    """
    chart = alt.Chart(plot_data_df).mark_bar().encode(
        x='group',
        y='uplift',
        color=alt.condition(
            alt.datum.uplift > 0,
            alt.value('green'),  # positive uplifts are green
            alt.value('red')  # negative uplifts are red
        )
    ).properties(
        width=alt.Step(40)  # controls the width of the bars
    )

    return chart

def create_scatter_plot_with_regression(df):
    """
    Creates a scatter plot with regression line using the df dataframe.

    Parameters:
    df (pandas.DataFrame): Dataframe containing the columns `treatment`, `conversion`, and `spend`.

    Returns:
    altair.vegalite.v4.api.Chart: An Altair chart displaying the scatter plot with regression line.
    """
    scatter = alt.Chart(df).mark_circle(size=60).encode(
        x=alt.X('treatment:Q', title='Treatment'),
        y=alt.Y('conversion:Q', title='Conversion'),
        color=alt.condition(
            alt.datum.treatment == 1,
            alt.value('blue'),
            alt.value('orange')
        )
    ).properties(
        width=500,
        height=400
    )

    # Add regression line
    regression = scatter.transform_regression(
        'treatment',
        'conversion',
        groupby=['spend']
    ).mark_line(color='red')

    return scatter + regression

def create_box_plot(df):
    """
    Creates a box plot using the df dataframe.

    Parameters:
    df (pandas.DataFrame): Dataframe containing the columns `group` and `conversion`.

    Returns:
    altair.vegalite.v4.api.Chart: An Altair chart displaying the box plot.
    """
    boxplot = alt.Chart(df).mark_boxplot().encode(
        x='group:N',
        y='conversion:Q'
    ).properties(
        width=alt.Step(40)  # controls the width of the boxes
    )

    return boxplot

def create_line_chart(df):
    """
    Creates a line chart using the df dataframe.

    Parameters:
    df (pandas.DataFrame): Dataframe containing the columns `date`, `conversion`, and `spend`.

    Returns:
    altair.vegalite.v4.api.Chart: An Altair chart displaying the line chart.
    """
    line_chart = alt.Chart(df).mark_line().encode(
        x='date:T',
        y='conversion:Q',
        color='spend:Q'
    )

    return line_chart

def create_bar_chart(df):
    # Group the data by offer and calculate the uplift
    grouped_df = df.groupby(['offer']).mean()[['treatment_group', 'conversion']]
    grouped_df['uplift'] = grouped_df['treatment_group'] - grouped_df['conversion']

    # Create a bar chart of the uplift by offer
    bar_chart = alt.Chart(grouped_df.reset_index()).mark_bar().encode(
        x='offer',
        y='uplift',
        tooltip=['offer', 'uplift']
    ).properties(
        width=700,
        height=400,
        title='Uplift by Offer'
    )

    return bar_chart

def create_uplift_cat_countplot(df):
    fig = plt.figure(figsize=(15,10))
    sns.set(style="whitegrid")
    ax = sns.catplot(x="segment", y="uplift", hue="treatment_group", data=df, kind="bar", palette="muted", height=6, aspect=1.5)
    ax.despine(left=True)
    ax.set_ylabels("Uplift")
    ax.set_xlabels("Segments")
    ax.set_xticklabels(rotation=90)
    plt.title('Uplift Across Segments')
    return fig


def decision_tree_plot(df):
    fig = plt.figure(figsize=(25,15))
    tree.plot_tree(df['clf'], filled=True, fontsize=10, feature_names=df.drop(['conversion', 'treatment_group', 'segment', 'proba'], axis=1).columns)
    return fig

def plot_qini_curve(qini_x, qini_y, y_test, trmnt_test, plot_data_df):
    fig = plt.figure(figsize=(10,8))
    plot_data_df['segment'] = plot_data_df['segment'].apply(lambda x: 'Treatment' if x == 1 else 'Control')
    plot_data_df['treatment_group'] = plot_data_df['treatment_group'].apply(lambda x: 'Treatment' if x == 1 else 'Control')
    plot_data_df['conversion'] = plot_data_df['conversion'].apply(lambda x: 'Yes' if x == 1 else 'No')
    plot_data_df['trmnt'] = y_test
    plot_data_df['y'] = trmnt_test
    qini_data = qini(qini_x, qini_y, y_test, trmnt_test)
    qini_chart = alt.Chart(qini_data).mark_line().encode(
        x=alt.X('sample_size', scale=alt.Scale(type='log')),
        y=alt.Y('uplift', scale=alt.Scale(domain=(0, 1))),
        tooltip=['sample_size', 'uplift'],
    ).properties(
        title='Qini Curve',
        width=700,
        height=400
    )

    return qini_chart

def explore_predicted_observations(df):
    columns = df.columns.tolist()
    features = columns[1:-2]
    category = st.selectbox("Select a Feature", features)
    category_df = df[['conversion', 'treatment_group', 'proba', category]].sort_values('proba', ascending=False)
    href = download_link(category_df, category)
    return category, href, category_df

def download_link(df, category):
    df = df.drop_duplicates(subset=category, keep="first")
    csv = df.to_csv(index=False)
    href = f'<a href="data:file/csv;base64,{base64.b64encode(csv.encode()).decode()}" download="{category}_predicted_observations.csv">Download {category} Predicted Observations CSV File</a>'
    return href

def main():
    st.set_page_config(page_title='Uplift Model', page_icon=':bar_chart:')
    st.title('Uplift Model - Campaign Analytics')
    
    tabs = {
        'Welcome': welcome_page,
        'Campaign Visualizations': grouped_count_plot,
        'Exploratory Data Analysis': exploratory_data_analysis,
        'Campaign Results': campaign_results_tab,
        'Uplift Segment': uplift_segment_tab
    }

    selected_tab = st.sidebar.radio('', list(tabs.keys()))

    tabs[selected_tab]()
def exploratory_data_analysis():
    st.write('Select analysis type:')
    analysis_type = st.selectbox('', ['Categorical Analysis', 'Numerical Analysis'])
    if analysis_type == 'Categorical Analysis':
        categorical_analysis()
    elif analysis_type == 'Numerical Analysis':
        numerical_analysis()


def campaign_results_tab():
    df, plot_data_df ,X_test_2, y_test, trmnt_test = campaign_results()

    # Create a selection box for the plots
    plot_options = [
        'Uplift Chart',
        'Scatter Plot',
        'Box Plot',
        'Line Chart',
        'Bar Chart'
    ]
    selected_plot = st.selectbox('Select a plot to display:', plot_options)

    # Create and display the selected plot
    plot_functions = {
        'Uplift Chart': create_uplift_chart,
        'Scatter Plot': create_scatter_plot_with_regression,
        'Box Plot': create_box_plot,
        'Line Chart': create_line_chart,
        'Bar Chart': create_bar_chart
    }
    plot_functions[selected_plot](df)


def uplift_segment_tab():
    plot_data_df, (qini_x, qini_y), y_test, trmnt_test = clean()

    plot_options = [
        'Uplift Histogram',
        'Uplift Count Plot',
        'Uplift Bar Plot',
        'Decision Tree Plot',
        'Qini Curve',
        'Uplift by Variable',
        'Explore and Download Predicted Observations'
    ]
    selected_plot = st.selectbox('Select a plot to display:', plot_options)

    plot_functions = {
        'Uplift Histogram': uplift_histogram,
        'Uplift Count Plot': uplift_count_plot,
        'Uplift Bar Plot': uplift_bar_plot,
        'Decision Tree Plot': decision_tree_plot,
        'Qini Curve': plot_qini_curve,
        'Uplift by Variable': create_uplift_cat_countplot,
        'Explore and Download Predicted Observations': explore_predicted_observations
    }
    plot_functions[selected_plot](plot_data_df, (qini_x, qini_y), y_test, trmnt_test)


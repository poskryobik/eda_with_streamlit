import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



total_df = pd.read_csv("datasets/total_df.csv", index_col=0)

st.title("EDA for bank")

# 1) DataFrame
st.markdown("**Dataframe**")
st.dataframe(total_df)


# 2) Histplots
st.markdown("**Charts**")
# Select box
param = st.selectbox(
     'What parameter is needed for the histogram?',
     ('AGE', 'TARGET', 'GENDER', 'CHILD_TOTAL', 'DEPENDANTS',
       'SOCSTATUS_WORK_FL', 'SOCSTATUS_PENS_FL', 'PERSONAL_INCOME',
       'LOAN_NUM_TOTAL', 'LOAN_NUM_CLOSED'))

fig, ax = plt.subplots()
sns.histplot(total_df[param], ax=ax)
plt.title(f"Histplot for {param}")
plt.grid()
st.pyplot(fig)

# 3) Heatmap
st.markdown("**Heatmap**")
corr = total_df.drop("AGREEMENT_RK", axis=1).corr()
fig_heat, ax_heat = plt.subplots(figsize=(10, 7))
sns.heatmap(corr, annot = True, cmap= 'Blues', ax=ax_heat)
plt.title("Correlation")
st.pyplot(fig_heat)



# 4) Target histplot
st.markdown("**Dependence on the target variable**")
param_t = st.selectbox(
     'What parameter is needed for visual?',
     ('AGE', 'GENDER', 'CHILD_TOTAL', 'DEPENDANTS',
       'SOCSTATUS_WORK_FL', 'SOCSTATUS_PENS_FL', 'PERSONAL_INCOME',
       'LOAN_NUM_TOTAL', 'LOAN_NUM_CLOSED'))

fig_target, ax_target = plt.subplots()
sns.histplot(total_df[total_df['TARGET'] == 0][param_t], 
             alpha=0.5, label='target: 0', 
             ax=ax_target)
sns.histplot(total_df[total_df['TARGET'] == 1][param_t], 
             alpha=0.5, label='target: 1', 
             ax=ax_target)
fig_target.legend()
plt.title(f"Histplot for {param_t}")
plt.grid()
st.pyplot(fig_target)

# 5) Numerical characteristics
st.markdown("**Numerical characteristics**")
st.table(total_df.drop("AGREEMENT_RK", axis=1).describe().T)


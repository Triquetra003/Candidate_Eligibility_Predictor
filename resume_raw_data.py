import ast

import pandas as pd
import matplotlib.pyplot as plt

resume_data=pd.read_csv('resume_data.csv');

print(resume_data.info());

# RangeIndex: 9544 entries, 0 to 9543
# Data columns (total 35 columns):
#  #   Column                               Non-Null Count  Dtype  
# ---  ------                               --------------  -----  
#  0   address                              784 non-null    str    
#  1   career_objective                     4740 non-null   str    
#  2   skills                               9488 non-null   str    
#  3   educational_institution_name         9460 non-null   str    
#  4   degree_names                         9460 non-null   str    
#  5   passing_years                        9460 non-null   str    
#  6   educational_results                  9460 non-null   str    
#  7   result_types                         9460 non-null   str    
#  8   major_field_of_studies               9460 non-null   str    
#  9   professional_company_names           9460 non-null   str    
#  10  company_urls                         9460 non-null   str    
#  11  start_dates                          9460 non-null   str    
#  12  end_dates                            9460 non-null   str    
#  13  related_skils_in_job                 9460 non-null   str    
#  14  positions                            9460 non-null   str    
#  15  locations                            9460 non-null   str    
#  16  responsibilities                     9544 non-null   str    
#  17  extra_curricular_activity_types      3426 non-null   str    
#  18  extra_curricular_organization_names  3426 non-null   str    
#  19  extra_curricular_organization_links  3426 non-null   str    
#  20  role_positions                       3426 non-null   str    
#  21  languages                            700 non-null    str    
#  22  proficiency_levels                   700 non-null    str    
#  23  certification_providers              2008 non-null   str    
#  24  certification_skills                 2008 non-null   str    
#  25  online_links                         2008 non-null   str    
#  26  issue_dates                          2008 non-null   str    
#  27  expiry_dates                         2008 non-null   str    
#  28  job_position_name                   9544 non-null   str    
#  29  educationaL_requirements             9544 non-null   str    
#  30  experiencere_requirement             8180 non-null   str    
#  31  age_requirement                      5457 non-null   str    
#  32  responsibilities.1                   9544 non-null   str    
#  33  skills_required                      7843 non-null   str    
#  34  matched_score                        9544 non-null   float64
# dtypes: float64(1), str(34)
# memory usage: 2.5 MB
# None

print(resume_data.describe());

#        matched_score
# count    9544.000000
# mean        0.660831
# std         0.167040
# min         0.000000
# 25%         0.583333
# 50%         0.683333
# 75%         0.793333
# max         0.970000

print(resume_data['matched_score'][resume_data['matched_score']==resume_data['matched_score'].max()]);
# 0.97
print(resume_data['matched_score'][resume_data['matched_score']==resume_data['matched_score'].min()]);
# 0.0
print(resume_data['matched_score'].mean());
# 0.6608308276326837

# DATA CLEANING
# -------------
print("Null values in each column:");
print(resume_data.isnull().sum());
# address                                8760
# career_objective                       4804
# skills                                   56
# educational_institution_name             84
# degree_names                             84
# passing_years                            84
# educational_results                      84
# result_types                             84
# major_field_of_studies                   84
# professional_company_names               84
# company_urls                             84
# start_dates                              84
# end_dates                                84
# related_skils_in_job                     84
# positions                                84
# locations                                84
# responsibilities                          0
# extra_curricular_activity_types        6118
# extra_curricular_organization_names    6118
# extra_curricular_organization_links    6118
# role_positions                         6118
# languages                              8844
# proficiency_levels                     8844
# certification_providers                7536
# certification_skills                   7536
# online_links                           7536
# issue_dates                            7536
# expiry_dates                           7536
# job_position_name                        0
# educationaL_requirements                  0
# experiencere_requirement               1364
# age_requirement                        4087
# responsibilities.1                        0
# skills_required                        1701
# matched_score                             0

# Duplicate rows in the dataset
resume_data.duplicated().sum();
# 0 - There are no duplicate rows in the dataset

# print(resume_data['age_requirement'].unique());
# <StringArray>
# [                    nan,    'Age 22 to 30 years',    'Age 25 to 40 years',
#  'Age at least 24 years', 'Age at least 28 years',    'Age 30 to 40 years',
#   'Age at most 40 years',    'Age 25 to 35 years',    'Age 20 to 35 years',
#  'Age at least 22 years',    'Age 18 to 30 years',  'Age at most 52 years',
#  'Age at least 30 years',    'Age 25 to 32 years',    'Age 25 to 30 years']
# Length: 15, dtype: str

# print(resume_data['experiencere_requirement'].unique());
# <StringArray>
# [   'At least 1 year', 'At least 5 year(s)',   'At least 3 years',
#        '1 to 3 years',   'At least 4 years',                  nan,
#        '2 to 5 years',   'At least 5 years',       '4 to 5 years',
#       '5 to 10 years',       '3 to 5 years',       '2 to 4 years',
#        '5 to 8 years',  'At least 15 years',       '3 to 7 years',
#    'At least 2 years',       '5 to 6 years',       '1 to 2 years']
# Length: 18, dtype: str

# Cleaning the 'skills_required' and 'skills' columns 
# Removing unnecessary characters,  whitespace and transformation to lower case 

split_skills_req=resume_data['skills_required'].str.split('\n')
strip_skills_req=split_skills_req.apply(lambda x: [i.strip().replace("•", "").lower() for i in x] if isinstance(x, list) else "No Requirements")
cleaned_skills_req=strip_skills_req;

split_skills=resume_data['skills'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
strip_skills=split_skills.apply(lambda x: [i.strip().replace("•", "").lower() for i in x] if isinstance(x, list) else "No Skills")
cleaned_skills=strip_skills;

# The cleaned anf normalized to lower case values are stored in new columns
resume_data['cleaned_skills_required']=cleaned_skills_req;
resume_data['cleaned_skills']=cleaned_skills;

# Cleaning the start_dates and end_dates columns to extract the year of experience

# Clean start_dates column and add it as new "cleaned_start_dates" column in the dataframe
start_dates=resume_data['start_dates'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x);
# Convert valid date formats to datetime and invalid date formats to NaT (Not a Time) 
start_dates=start_dates.apply(lambda x: [pd.to_datetime(i, errors='coerce') if i!="N/A" else pd.NaT for i in x] if isinstance(x, list) else x);
resume_data['cleaned_start_dates']=start_dates;


# Clean end_dates column and add it as new "cleaned_end_dates" column in the dataframe
end_dates=resume_data['end_dates'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x);
# Convert valid date formats to datetime and invalid date formats to NaT (Not a Time)
# Convert "till date", "current" and "ongoing" to a single value to represent ongoing employment
for i in range(len(end_dates)):
    if not isinstance(end_dates[i], list):
        end_dates[i]=pd.NaT;
    else:
        for j in range(len(end_dates[i])):
            if end_dates[i][j]==None:
                end_dates[i][j]=pd.NaT;
            else:
                end_dates[i][j]=end_dates[i][j].lower().strip();
                if end_dates[i][j]=="till date" or end_dates[i][j]=="current" or end_dates[i][j]=="ongoing":
                    # Latest date in dataset for end_Dates / max date from end dates = 2023-10-01
                    end_dates[i][j]=pd.to_datetime("2023-10-01");
                else:
                    end_dates[i][j]=pd.to_datetime(end_dates[i][j], errors='coerce');
resume_data['cleaned_end_dates']=end_dates;

# Experience calculation based on cleaned start_dates and end_dates columns
# Employment intervals with chronologically invalid dates were excluded from experience calculation.
# for i in range(len(start_dates)):
#     if isinstance(start_dates[i], list) and isinstance(end_dates[i], list):
#         for j in range(len(start_dates[i])):
#             if pd.notna(start_dates[i][j]) and pd.notna(end_dates[i][j]):
#                 if start_dates[i][j] > end_dates[i][j]:
#                     print("Invalid:", i, j)

# checking overlaps










# DATA VISUALIZATION
# ------------------
# resume_data['matched_score'].hist(bins=20);
# plt.show();

# resume_data.plot.scatter(x='experience', y='matched_score');
# plt.show();
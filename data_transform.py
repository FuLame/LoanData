import numpy as np

#dummy variables
home_ownership_dict = {'OWN':0, 'NONE':1, 'OTHER':2,
                       'MORTGAGE':3, 'RENT':4, 'ANY':5}
application_type_dict = {'INDIVIDUAL':0, 'JOINT':1}
initial_list_status_dict = {'w':0, 'f':1}
verification_status_dict = {'Verified':0, 'Not Verified':1,
                            'Source Verified':2}
pymnt_plan_dict = {'y':1, 'n':0}


label_dict = {'Default':0, 'Charged Off':0,'Late (31-120 days)':0,
              'Does not meet the credit policy. Status:Charged Off':0,
              'Late (16-30 days)':1, 'Issued':1, 'Current':1,
              'Does not meet the credit policy. Status:Fully Paid':1,
              'In Grace Period':1, 'Fully Paid':1}


def transform_data(dataFrame):

    dataFrame.replace('n/a', np.nan, inplace=True)

    #Add labels
    dataFrame['label'] = dataFrame.loan_status.replace(label_dict)

    #Drop columns with majority of NaNs
    dataFrame = dataFrame.dropna(axis=1, thresh=int(len(dataFrame) / 2))

    #Drop descriptive columns
    dataFrame.drop(['id', 'member_id', 'emp_title', 'url',
                    'title', 'purpose', 'loan_status','policy_code'],
                   axis=1, inplace=True)
    #Temporary
    dataFrame.drop(['next_pymnt_d','addr_state',
                    'grade', 'sub_grade'],
                   axis=1, inplace=True)

    #Drop redundant columns (with high correlation to other)
    dataFrame.drop(['funded_amnt','funded_amnt_inv',
                    'total_pymnt_inv','out_prncp_inv'],
                   axis=1, inplace=True)

    #Create manual dummy variables
    dataFrame.home_ownership.replace(home_ownership_dict,inplace=True)
    dataFrame.home_ownership = dataFrame.home_ownership.astype(int)

    dataFrame.application_type.replace(application_type_dict, inplace=True)
    dataFrame.application_type = dataFrame.application_type.astype(int)

    dataFrame.initial_list_status.replace(initial_list_status_dict, inplace=True)
    dataFrame.initial_list_status = dataFrame.initial_list_status.astype(int)

    dataFrame.verification_status.replace(verification_status_dict, inplace=True)
    dataFrame.verification_status = dataFrame.verification_status.astype(int)

    dataFrame.pymnt_plan.replace(pymnt_plan_dict, inplace=True)
    dataFrame.pymnt_plan = dataFrame.pymnt_plan.astype(int)

    dataFrame.emp_length.fillna(value=0, inplace=True)
    dataFrame.emp_length.replace(to_replace='[^0-9]+', value='', inplace=True, regex=True)
    dataFrame.emp_length = dataFrame.emp_length.astype(int)

    dataFrame.term.replace(to_replace='[^0-9]+', value='', inplace=True, regex=True)
    dataFrame.term = dataFrame.term.astype(int)

    dataFrame.zip_code.replace(to_replace='[^0-9]+', value='', inplace=True, regex=True)
    dataFrame.zip_code = dataFrame.zip_code.astype(int)

    dataFrame.last_pymnt_d.replace(to_replace='[^0-9]+', value='', inplace=True, regex=True)
    dataFrame.last_pymnt_d.fillna(value=0, inplace=True)
    dataFrame.last_pymnt_d = dataFrame.last_pymnt_d.astype(int)

    dataFrame.issue_d.replace(to_replace='[^0-9]+', value='', inplace=True, regex=True)
    dataFrame.issue_d.fillna(value=0, inplace=True)
    dataFrame.issue_d = dataFrame.issue_d.astype(int)

    dataFrame.last_credit_pull_d.replace(to_replace='[^0-9]+', value='', inplace=True, regex=True)
    dataFrame.last_credit_pull_d.fillna(value=0, inplace=True)
    dataFrame.last_credit_pull_d = dataFrame.last_credit_pull_d.astype(int)

    dataFrame.earliest_cr_line.replace(to_replace='[^0-9]+', value='', inplace=True, regex=True)
    dataFrame.earliest_cr_line.fillna(value=0, inplace=True)
    dataFrame.earliest_cr_line = dataFrame.earliest_cr_line.astype(int)

    #Fill NaNs with proper values according to distribution
    #Log tranformation restores normal distribution -> mean
    dataFrame['log_tot_cur_bal'] = np.log1p(dataFrame.tot_cur_bal)
    dataFrame.drop(['tot_cur_bal'], axis=1, inplace=True)

    dataFrame['log_total_rev_hi_lim'] = np.log1p(dataFrame.total_rev_hi_lim)
    dataFrame.drop(['total_rev_hi_lim'], axis=1, inplace=True)

    # Log tranformation is not sufficient -> median
    dataFrame.tot_coll_amt.fillna(value=dataFrame.tot_coll_amt.median(), inplace=True)

    #Only few values are missing -> NaNs replaced with mean values
    dataFrame.fillna(value=dataFrame.mean(), inplace=True)

    # ADVANCED DATA TRANSFORM
    '''
    # Log tranformation restores normal distribution -> mean
    dataFrame['log_annual_inc'] = np.log1p(dataFrame.annual_inc)
    dataFrame.drop(['annual_inc'], axis=1, inplace=True)

    dataFrame['log_revol_bal'] = np.log1p(dataFrame.revol_bal)
    dataFrame.drop(['revol_bal'], axis=1, inplace=True)

    dataFrame['log_total_rec_int'] = np.log1p(dataFrame.total_rec_int)
    dataFrame.drop(['total_rec_int'], axis=1, inplace=True)

    dataFrame['log_total_rec_prncp'] = np.log1p(dataFrame.total_rec_prncp)
    dataFrame.drop(['total_rec_prncp'], axis=1, inplace=True)

    dataFrame['log_last_pymnt_amnt'] = np.log1p(dataFrame.last_pymnt_amnt)
    dataFrame.drop(['last_pymnt_amnt'], axis=1, inplace=True)

    dataFrame['log_dti'] = np.log1p(dataFrame.dti)
    dataFrame.drop(['dti'], axis=1, inplace=True)
    '''

    return dataFrame


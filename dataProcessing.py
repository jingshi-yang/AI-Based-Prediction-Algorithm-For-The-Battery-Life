import glob

import numpy as np
import pandas as pd


def read_data(data_list, data_path):
    battery = {}
    for name in data_list:
        print('Loading dataset' + name + ' ...')
        path = glob.glob(data_path + name + '/*.xlsx')
        dates = []
        for p in path:
            df = pd.read_excel(p, sheet_name=1)
            print('Loading ' + str(p) + ' ...')
            dates.append(df['Date_Time'][0])
            idx = np.array(path)[idx]
            path_sorted = np.array(path)[idx]

            count = 0
            discharge_capacities = []
            health_indicator = []
            ir = []
            CC_charge_time = []
            CV_charge_time = []
            for p in path_sorted:
                df = pd.read_excel(p, sheet_name=1)
                print('Loading ' + str(p) + '...')
                cycles = list(set(df['Cycle_Index']))
                for c in cycles:
                    df_lim = df[df['Cycle_Index'] == c]
                    # Charging
                    df_c = df_lim[(df_lim['Step_Index'] == 2) | (df_lim['Step_Index'] == 4)]
                    c_v = df_c['Voltage(V)']
                    c_c = df_c['Current(A)']
                    c_t = df_c['Test_Time(s)']
                    # CC or CV
                    df_cc = df_lim[df_lim['Step_Index'] == 2]
                    df_cv = df_lim[df_lim['Step_Index'] == 4]
                    CC_charge_time.append(np.max(df_cc['Test_Tims(s)']) - np.min(df_cc['Test_Time(s)']))
                    CV_charge_time.append(np.max(df_cv['Test_Tims(s)']) - np.min(df_cc['Test_Time(s)']))
                    # Discharging
                    df_d = df_lim[df_lim['Step_Index'] == 7]
                    d_v = df_d['Voltage(V)']
                    d_c = df_d['Current(A)']
                    d_t = df_d['Test_Time(s)']
                    d_ir = df_d['Internal_Resistance(Ohm)']

                    if len(list(d_c)) != 0:
                        time_diff = np.diff(list(d_t))
                        d_c = np.array(list(d_c))[1:]
                        discharge_capacity = time_diff * d_c / 3600
                        discharge_capacity = [np.sum(discharge_capacity[:n])
                                              for n in range(discharge_capacity.shape[0])]
                        discharge_capacities.append(-1 * discharge_capacity[-1])

                        dec = np.abs(np.array(d_v) - 3.8)[1:]
                        start = np.array(discharge_capacity)[np.argmin(dec)]
                        dec = np.abs(np.array(d_v) - 3.4)[1:]
                        end = np.array(discharge_capacity)[np.argmin(dec)]
                        health_indicator.append(-1 * (end - start))

                        ir.append(np.mean(np.array(d_ir)))
                        count += 1

            discharge_capacities = np.array(discharge_capacities)
            health_indicator = np.array(health_indicator)
            ir = np.array(ir)
            CV_charge_time = np.array(CV_charge_time)
            CC_charge_time = np.array(CC_charge_time)

            df_result = pd.DataFrame({'cycle': np.linspace(1, idx.shape[0], idx.shape[0]),
                                      'capacity': discharge_capacities[idx],
                                      'SOH': health_indicator[idx],
                                      'resistance': ir[idx],
                                      'CCCT': CC_charge_time[idx],
                                      'CVCT': CV_charge_time[idx]
                                      })
            battery[name] = df_result
    return battery

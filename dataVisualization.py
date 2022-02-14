import numpy as np
import matplotlib.pyplot as plt
a= np.load('CALCE.npy',allow_pickle=True);
a  = a.item();
print(a);

plt.figure(1);
plt.scatter(a['CS2_35'].cycle,a['CS2_35'].SoH,label = 'CS2_35');
plt.scatter(a['CS2_36'].cycle,a['CS2_36'].SoH,label = 'CS2_36');
plt.scatter(a['CS2_37'].cycle,a['CS2_37'].SoH,label = 'CS2_37');
plt.scatter(a['CS2_38'].cycle,a['CS2_38'].SoH,label = 'CS2_38');
plt.title('SoH Degradation')
plt.xlabel('Cycle',fontsize ='large');
plt.ylabel('SoH',fontsize = 'large');
plt.legend(loc='upper right', shadow=True, fontsize='large');
plt.show();


plt.figure(2);
plt.scatter(a['CS2_35'].cycle,a['CS2_35'].capacity,label = 'CS2_35');
plt.scatter(a['CS2_36'].cycle,a['CS2_36'].capacity,label = 'CS2_36');
plt.scatter(a['CS2_37'].cycle,a['CS2_37'].capacity,label = 'CS2_37');
plt.scatter(a['CS2_38'].cycle,a['CS2_38'].capacity,label = 'CS2_38');
plt.title('Capacity Degradation')
plt.xlabel('Cycle',fontsize ='large');
plt.ylabel('Capacity',fontsize = 'large');
plt.legend(loc='upper right', shadow=True, fontsize='large');
plt.show();

plt.figure(3);
plt.scatter(a['CS2_35'].resistance,a['CS2_35'].capacity,label = 'CS2_35');
plt.scatter(a['CS2_36'].resistance,a['CS2_36'].capacity,label = 'CS2_36');
plt.scatter(a['CS2_37'].resistance,a['CS2_37'].capacity,label = 'CS2_37');
plt.scatter(a['CS2_38'].resistance,a['CS2_38'].capacity,label = 'CS2_38');
plt.title('resistance and capacity')
plt.xlabel('resistance',fontsize ='large');
plt.ylabel('Capacity',fontsize = 'large');
plt.legend(loc='upper right', shadow=True, fontsize='large');
plt.show();


plt.figure(4);
plt.scatter(a['CS2_35'].cycle,a['CS2_35'].resistance,label = 'CS2_35');
plt.scatter(a['CS2_36'].cycle,a['CS2_36'].resistance,label = 'CS2_36');
plt.scatter(a['CS2_37'].cycle,a['CS2_37'].resistance,label = 'CS2_37');
plt.scatter(a['CS2_38'].cycle,a['CS2_38'].resistance,label = 'CS2_38');
plt.title('resistance change after discharge cycles')
plt.xlabel('cycle',fontsize ='large');
plt.ylabel('resistance',fontsize = 'large');
plt.legend(loc='upper right', shadow=True, fontsize='large');
plt.show();





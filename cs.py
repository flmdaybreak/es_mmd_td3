import numpy as np



arr = np.array([9,8,7,6,5,4,3,2])
b=np.arange(len(arr))
c=b[arr>3]
print(c)
mmd_dist_list=[1,2,0,5,6,0,0,0,0,0]
scores_mmd_dist_list = np.array(mmd_dist_list)
rank_mmd = np.arange(len(scores_mmd_dist_list))
rank_mmds = rank_mmd[scores_mmd_dist_list > 0]
le = len(rank_mmds)
scores_mmd_dist_list *= -1
idx_sorted = np.argsort(scores_mmd_dist_list)
arr1 = idx_sorted[1:le]
#arr1=arr1[::-1]
arr3 = arr[0]

index = np.argwhere(arr ==arr3)
a1 = np.delete(arr,index )
arr2 = arr[le:]
idx_sorted2 = np.append(arr3,arr1)
idx_sorted1 = np.append(idx_sorted2,arr2)
print(idx_sorted)
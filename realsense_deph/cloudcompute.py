import cloudComPy as cc
cc.initCC()

cloud = cc.loadPointCloud('data/point_cloud (1).pcd') 
res=cc.computeCurvature(cc.CurvatureType.GAUSSIAN_CURV, 0.05, [cloud]) 


from django.urls import path
from . import views

urlpatterns = [
    path('', views.home_redirect),
    path('explore/', views.data_explore, name='data_explore'),
    path('predict/', views.predict_page, name='predict'),
    path('dashboard/', views.system_dashboard, name='system_dashboard'),
    path('api/explore/stats/', views.api_explore_stats),
    path('api/explore/congestion-top10/', views.api_congestion_top10),
    path('api/explore/topology/', views.api_topology),
    path('api/explore/topology3d/', views.api_topology_3d),
    path('api/explore/timeseries/', views.api_timeseries),
    path('api/explore/patterns/', views.api_patterns),
    path('api/explore/polar-period/', views.api_polar_period),
    path('api/predict/run/', views.api_predict),
]

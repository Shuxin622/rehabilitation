"""djangoProject URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.conf.urls import url
from django.urls import path
from . import views

urlpatterns = [
    path('index/', views.index, name='index'),
    path('tt/', views.tt, name='tt'),

    path('video/', views.video, name='video'),
    path('<int:num>/video/', views.video, name='video1'),
    path('<int:num>/<int:back>/video/', views.video, name='video2'),
    path('videoUpload/', views.video_upload, name='videoUpload'),
    path('joint/', views.joint, name='joint'),
    path('<int:num>/joint/', views.joint, name='joint1'),
    path('<int:num>/<int:track>/joint/', views.joint, name='joint2'),
    path('<int:num>/<int:track>/<int:back>/joint/', views.joint, name='joint3'),
    path('jointTrack/', views.joint_track, name='jointTrack'),
    path('hingePoint/', views.hinge_point, name='hingePoint'),
    path('<int:num>/hingePoint/', views.hinge_point, name='hingePoint1'),
    path('<int:num>/<int:back>/hingePoint/', views.hinge_point, name='hingePoint2'),
    path('mechanicalDesign/', views.mechanical_design, name='mechanicalDesign'),
    path('mechanicalDesignForce/', views.mechanical_design_force, name='mechanicalDesign_force'),
    path('PositionSyn/', views.PositionSyn, name='PositionSyn'),
    path('<int:num>/PositionSyn/', views.PositionSyn, name='PositionSyn1'),
    path('<int:num>/<int:status_video>/PositionSyn/', views.PositionSyn, name='PositionSyn2'),
    path('ratioindex/',views.ratioindex,name='ratioindex'),
    path('para/',views.para,name='para'),
    path('<int:num>/<int:resultIndex>/para/',views.para,name='para1'),
    path('<int:num>/<int:flag_video>/<int:resultIndex>/para/', views.para, name='para2'),
    path('ForceSyn/',views.ForceSyn,name='ForceSyn'),
    path('<int:num>/ForceSyn/', views.ForceSyn, name='ForceSyn1'),
    path('ratioindexForce/',views.ratioindexForce,name='ratioindexForce'),
    path('<int:num>/<int:resultIndex>/paraForce/',views.paraForce,name='paraForce1'),

]


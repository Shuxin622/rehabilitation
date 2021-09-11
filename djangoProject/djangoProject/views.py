from django.http import HttpResponse
from django.shortcuts import render
import base64
from django.http import HttpResponse, HttpResponseRedirect, StreamingHttpResponse
from django.urls import reverse
import shutil
import os
import cv2
import numpy as np
import math
from pathlib import Path
import scipy.stats as stats

from kcf import kcftracker
from mechanical import mechanicalDesign
from mechanical import nBar
from mechanical import positionForce

nBar_design = nBar.nBar()
force_design = positionForce.positionForce()



def video(request, num=0, back=0):
    return render(request, 'video.html', {'num': num, 'back': back})


def video_upload(request):
    shutil.rmtree('./static/images/video')
    os.mkdir('./static/images/video')
    num = int(request.POST['num'])
    for i in range(num):
        img = request.POST['img' + str(i)].split(',')[1]
        img = base64.b64decode(img)
        with open('./static/images/video/%d.jpg' % i, 'wb') as f:
            f.write(img)

    return HttpResponseRedirect(reverse('video1',args=(num,)))


def joint(request, num=0, track=0, back=0):
    x = 160
    y = 120
    width = 320
    height = 240
    if track == 1:
        with open('./static/images/track_video/area.txt', 'r') as f:
            line = f.read()
            line = line.split(' ')
            x = int(line[0])
            y = int(line[1])
            width = int(line[2])
            height = int(line[3])
    return render(request, 'joint.html', {'num': num, 'track': track, 'back': back,
                                                'x': x, 'y': y, 'width': width, 'height': height})


def joint_track(request):
    shutil.rmtree('./static/images/track_video')
    os.mkdir('./static/images/track_video')
    centers = []
    x_ = int(request.POST['x'])
    y_ = int(request.POST['y'])
    width_ = int(request.POST['width'])
    height_ = int(request.POST['height'])
    # print(x_, y_, width_, height_)


    with open('./static/images/track_video/area.txt', 'w+') as f:
        f.write("%d %d %d %d" % (x_, y_, width_, height_))

    tracker = kcftracker.KCFTracker()
    img = cv2.imread('./static/images/video/0.jpg')
    tracker.init([x_, y_, width_, height_], img)
    num = 0
    while img is not None:
        _, bounding_box = tracker.update(img)
        x = int(bounding_box[0])

        y = int(bounding_box[1])
        width = int(bounding_box[2])
        height = int(bounding_box[3])
        centers.append((x + width // 2, y + height //2))
        cv2.rectangle(img, (x, y), (x + width, y + height), (255, 0, 0), 2)
        for i in range(1, num):
            cv2.line(img, centers[i - 1], centers[i], (0, 0, 255), 2)
        cv2.imwrite('./static/images/track_video/%d.jpg' % num, img)

        num += 1
        img = cv2.imread('./static/images/video/%d.jpg' % num)

    with open('./static/images/track_video/trace.txt', 'w+') as f:
        for center in centers:
            f.write("%d %d\n" % (center[0], center[1]))

    return HttpResponseRedirect(reverse('joint2', args=(num, 1)))


def hinge_point(request, num1=0, back1=0):
    return render(request, 'joint2', {'num1': num1, 'back1': back1})


def coordinate_process(x, y, offset_x=80, offset_y=60):
    x = int(x) + offset_x
    y = int(-y) + offset_y
    return (x,y)

#  Position
def mechanical_design(request):
    shutil.rmtree('./static/images/previewImg')
    os.mkdir('./static/images/previewImg')
    x = int(request.POST['pivotx'])
    y = int(request.POST['pivoty'])
    maxlength = request.POST['maxLength']
    minlength = request.POST['minLength']

    width = int(request.POST['pivotwidth'])
    height = int(request.POST['pivotheight'])

    status_pole = int(request.POST['status_pole'])
    expansion = request.POST['expansion']
    # print(expansion)
    # timenum = int(request.POST['timenum'])
    timenum = 1

    motion = np.loadtxt('./static/images/track_video/trace.txt')
    nBar_design.clear()
    if status_pole & 1:
        nBar_design.Motion(motion, 2, x, y, width, height,maxlength,minlength,timenum)
    if status_pole >> 1 & 1:
        nBar_design.Motion(motion, 3, x, y, width, height,maxlength,minlength,timenum)
    if status_pole >> 2 & 1:
        nBar_design.Motion(motion, 4, x, y, width, height,maxlength,minlength,timenum)

    num = nBar_design.numPoints
    r_hole = 5
    r_role = 15
    r_wheel = [[25, 30], [35, 40], [48, 58]]
    color_black = (0, 0, 0)
    color_wheel = [(79, 79, 47), (139, 134, 0), (79, 79, 47)]
    color_hole = (255, 255, 255)
    thickness_border = 1
    color_pole = [(66, 121, 238), (105, 105, 105), (66, 121, 238), (105, 105, 105)]
    num_mechanical = len(nBar_design.role_num)

    for i in range(num_mechanical):
        R0x = nBar_design.R0x[i]
        R0y = nBar_design.R0y[i]
        num_role = nBar_design.role_num[i]
        phase = nBar_design.phase[i]
        length = nBar_design.length[i]
        with open('./static/model/%d.txt' % i, 'w') as f:
            f.write('origin: ' + str(R0x) + ' ' + str(R0y) + '\n')
            f.write('pole num: ' + str(num_role) + '\n')
            length_s = ""
            for l in length:
                length_s += str(l) + ' '
            f.write('length: ' + length_s + '\n')

            for j in range(num_role):
                phase_s = 'pole %d phase: ' % (j + 1)
                for p in phase[j]:
                    phase_s += str(p) + ' '
                f.write(phase_s + '\n')

        Rx = []
        Ry = []
        prev_x = R0x
        prev_y = R0y
        cos_value = []
        sin_value = []
        for j in range(num_role):
            cos_ = math.cos(math.radians(phase[j][0]))
            sin_ = math.sin(math.radians(phase[j][0]))
            prev_x += length[j] * cos_
            prev_y += length[j] * sin_
            cos_value.append(cos_)
            sin_value.append(sin_)
            Rx.append(prev_x)
            Ry.append(prev_y)

        belts = []
        for j in range(num_role - 1):
            r1 = r_wheel[num_role - 2 - j][1] - 1
            r2 = r_wheel[num_role - 2 - j][0] - 1
            r = r_wheel[num_role - 2 - j][1] - r_wheel[num_role - 2 - j][0]
            l = length[j]
            delta_x1 = r1 * r / l
            delta_y1 = math.sqrt(r1**2 - delta_x1**2)
            delta_x2 = r2 * r / l
            delta_y2 = math.sqrt(r2**2 - delta_x2**2)

            belt = [[[delta_x1, delta_y1], [l + delta_x2, delta_y2]],
                    [[delta_x1, -delta_y1], [l + delta_x2, -delta_y2]]]
            belts.append(belt)

        Rx.insert(0, R0x)
        Ry.insert(0, R0y)

        offset_x = int(300 - ((min(Rx) + max(Rx)) / 2))
        offset_y = int(300 + ((min(Ry) + max(Ry)) / 2))

        Rx_ = Rx[:]
        Ry_ = Ry[:]
        for j in range(num_role + 1):
            Rx_[j], Ry_[j] = coordinate_process(Rx_[j], Ry_[j], offset_x, offset_y)

        img = np.ones([600, 600, 3], np.uint8) * 255

        for j in range(num_role - 1):
            for belt in belts[j]:
                x1 = belt[0][0] * cos_value[j] - belt[0][1] * sin_value[j] + Rx[j]
                y1 = belt[0][0] * sin_value[j] + belt[0][1] * cos_value[j] + Ry[j]
                x2 = belt[1][0] * cos_value[j] - belt[1][1] * sin_value[j] + Rx[j]
                y2 = belt[1][0] * sin_value[j] + belt[1][1] * cos_value[j] + Ry[j]
                img = cv2.line(img, coordinate_process(x1, y1, offset_x, offset_y),
                               coordinate_process(x2, y2, offset_x, offset_y),
                               color_wheel[j], 2, lineType=cv2.LINE_AA)
            img = cv2.circle(img, (Rx_[j], Ry_[j]), r_wheel[num_role - 2 - j][1], color_wheel[j], -1,
                             lineType=cv2.LINE_AA)
            img = cv2.circle(img, (Rx_[j + 1], Ry_[j + 1]), r_wheel[num_role - 2 - j][0], color_wheel[j], -1,
                             lineType=cv2.LINE_AA)

        for j in range(num_role):

            img = cv2.circle(img, (Rx_[j], Ry_[j]), r_role, color_pole[j], -1, lineType=cv2.LINE_AA)
            img = cv2.circle(img, (Rx_[j], Ry_[j]), r_role + thickness_border, color_black, thickness_border,
                             lineType=cv2.LINE_AA)

            if j + 1 == num_role:
                img = cv2.circle(img, (Rx_[j + 1], Ry_[j + 1]), r_role, color_pole[j], -1, lineType=cv2.LINE_AA)
                img = cv2.circle(img, (Rx_[j + 1], Ry_[j + 1]), r_role + thickness_border, color_black, thickness_border,
                                 lineType=cv2.LINE_AA)

            img = cv2.line(img, (Rx_[j], Ry_[j]), (Rx_[j + 1], Ry_[j + 1]),
                           color_pole[j], 2 * r_role, lineType=cv2.LINE_AA)
            img = cv2.circle(img, (Rx_[j], Ry_[j]), r_hole, color_hole, -1, lineType=cv2.LINE_AA)
            img = cv2.circle(img, (Rx_[j], Ry_[j]), r_hole + 1, color_black, 1, lineType=cv2.LINE_AA)
            x1 = Rx[j] - sin_value[j] * (r_role + thickness_border)
            y1 = Ry[j] + cos_value[j] * (r_role + thickness_border)
            x2 = x1 + Rx[j + 1] - Rx[j]
            y2 = y1 + Ry[j + 1] - Ry[j]
            img = cv2.line(img, coordinate_process(x1, y1, offset_x, offset_y),
                           coordinate_process(x2, y2, offset_x, offset_y),
                           color_black, thickness_border, lineType=cv2.LINE_AA)

            x1 = Rx[j] + sin_value[j] * (r_role + thickness_border)
            y1 = Ry[j] - cos_value[j] * (r_role + thickness_border)
            x2 = x1 + Rx[j + 1] - Rx[j]
            y2 = y1 + Ry[j + 1] - Ry[j]
            img = cv2.line(img, coordinate_process(x1, y1, offset_x, offset_y),
                           coordinate_process(x2, y2, offset_x, offset_y),
                           color_black, thickness_border, lineType=cv2.LINE_AA)

        cv2.imwrite('./static/images/previewImg/' + str(i) + '.jpg', img)

    return HttpResponseRedirect(reverse('PositionSyn1', args=(num, )))


def mechanical_design_force(request):
    shutil.rmtree('./static/images/previewImgForce')
    os.mkdir('./static/images/previewImgForce')
    x = int(request.POST['pivotx'])
    y = int(request.POST['pivoty'])
    width = int(request.POST['pivotwidth'])
    height = int(request.POST['pivotheight'])
    maxlength = request.POST['maxLength']
    minlength = request.POST['minLength']
    status_pole = int(request.POST['status_pole'])
    timenum = 1
    # expansion = int(request.POST['expansion'])
    # print(expansion)
    # timenum = int(request.POST['timenum'])
    torque1 = request.POST['torque1']
    torque2 = request.POST['torque2']
    torque3 = request.POST['torque3']
    torque4 = request.POST['torque4']
    torque5 = request.POST['torque5']

    motion = np.loadtxt('./static/images/track_video/trace.txt')
    force_design.clear()
    if status_pole & 1:
        force_design.Motion(motion, 2, x, y, width, height,maxlength,minlength,timenum,torque1,torque2,torque3,torque4,torque5)
    if status_pole >> 1 & 1:
        force_design.Motion(motion, 3, x, y, width, height,maxlength,minlength,timenum,torque1,torque2,torque3,torque4,torque5)
    if status_pole >> 2 & 1:
        force_design.Motion(motion, 4, x, y, width, height,maxlength,minlength,timenum,torque1,torque2,torque3,torque4,torque5)

    num = force_design.numPoints
    r_hole = 5
    r_role = 15
    r_wheel = [[25, 30], [35, 40], [48, 58]]
    color_black = (0, 0, 0)
    color_wheel = [(79, 79, 47), (139, 134, 0), (79, 79, 47)]
    color_hole = (255, 255, 255)
    thickness_border = 1
    color_pole = [(66, 121, 238), (105, 105, 105), (66, 121, 238), (105, 105, 105)]
    num_mechanical = len(force_design.role_num)

    for i in range(num_mechanical):
        R0x = force_design.R0x[i]
        R0y = force_design.R0y[i]
        num_role = force_design.role_num[i]
        phase = force_design.phase[i]
        length = force_design.length[i]
        with open('./static/model/%d.txt' % i, 'w') as f:
            f.write('origin: ' + str(R0x) + ' ' + str(R0y) + '\n')
            f.write('pole num: ' + str(num_role) + '\n')
            length_s = ""
            for l in length:
                length_s += str(l) + ' '
            f.write('length: ' + length_s + '\n')

            for j in range(num_role):
                phase_s = 'pole %d phase: ' % (j + 1)
                for p in phase[j]:
                    phase_s += str(p) + ' '
                f.write(phase_s + '\n')

        Rx = []
        Ry = []
        prev_x = R0x
        prev_y = R0y
        cos_value = []
        sin_value = []
        for j in range(num_role):
            cos_ = math.cos(math.radians(phase[j][0]))
            sin_ = math.sin(math.radians(phase[j][0]))
            prev_x += length[j] * cos_
            prev_y += length[j] * sin_
            cos_value.append(cos_)
            sin_value.append(sin_)
            Rx.append(prev_x)
            Ry.append(prev_y)

        belts = []
        for j in range(num_role - 1):
            r1 = r_wheel[num_role - 2 - j][1] - 1
            r2 = r_wheel[num_role - 2 - j][0] - 1
            r = r_wheel[num_role - 2 - j][1] - r_wheel[num_role - 2 - j][0]
            l = length[j]
            delta_x1 = r1 * r / l
            delta_y1 = math.sqrt(r1**2 - delta_x1**2)
            delta_x2 = r2 * r / l
            delta_y2 = math.sqrt(r2**2 - delta_x2**2)

            belt = [[[delta_x1, delta_y1], [l + delta_x2, delta_y2]],
                    [[delta_x1, -delta_y1], [l + delta_x2, -delta_y2]]]
            belts.append(belt)

        Rx.insert(0, R0x)
        Ry.insert(0, R0y)

        offset_x = int(300 - ((min(Rx) + max(Rx)) / 2))
        offset_y = int(300 + ((min(Ry) + max(Ry)) / 2))

        Rx_ = Rx[:]
        Ry_ = Ry[:]
        for j in range(num_role + 1):
            Rx_[j], Ry_[j] = coordinate_process(Rx_[j], Ry_[j], offset_x, offset_y)

        img = np.ones([600, 600, 3], np.uint8) * 255

        for j in range(num_role - 1):
            for belt in belts[j]:
                x1 = belt[0][0] * cos_value[j] - belt[0][1] * sin_value[j] + Rx[j]
                y1 = belt[0][0] * sin_value[j] + belt[0][1] * cos_value[j] + Ry[j]
                x2 = belt[1][0] * cos_value[j] - belt[1][1] * sin_value[j] + Rx[j]
                y2 = belt[1][0] * sin_value[j] + belt[1][1] * cos_value[j] + Ry[j]
                img = cv2.line(img, coordinate_process(x1, y1, offset_x, offset_y),
                               coordinate_process(x2, y2, offset_x, offset_y),
                               color_wheel[j], 2, lineType=cv2.LINE_AA)
            img = cv2.circle(img, (Rx_[j], Ry_[j]), r_wheel[num_role - 2 - j][1], color_wheel[j], -1,
                             lineType=cv2.LINE_AA)
            img = cv2.circle(img, (Rx_[j + 1], Ry_[j + 1]), r_wheel[num_role - 2 - j][0], color_wheel[j], -1,
                             lineType=cv2.LINE_AA)

        for j in range(num_role):

            img = cv2.circle(img, (Rx_[j], Ry_[j]), r_role, color_pole[j], -1, lineType=cv2.LINE_AA)
            img = cv2.circle(img, (Rx_[j], Ry_[j]), r_role + thickness_border, color_black, thickness_border,
                             lineType=cv2.LINE_AA)

            if j + 1 == num_role:
                img = cv2.circle(img, (Rx_[j + 1], Ry_[j + 1]), r_role, color_pole[j], -1, lineType=cv2.LINE_AA)
                img = cv2.circle(img, (Rx_[j + 1], Ry_[j + 1]), r_role + thickness_border, color_black, thickness_border,
                                 lineType=cv2.LINE_AA)

            img = cv2.line(img, (Rx_[j], Ry_[j]), (Rx_[j + 1], Ry_[j + 1]),
                           color_pole[j], 2 * r_role, lineType=cv2.LINE_AA)
            img = cv2.circle(img, (Rx_[j], Ry_[j]), r_hole, color_hole, -1, lineType=cv2.LINE_AA)
            img = cv2.circle(img, (Rx_[j], Ry_[j]), r_hole + 1, color_black, 1, lineType=cv2.LINE_AA)
            x1 = Rx[j] - sin_value[j] * (r_role + thickness_border)
            y1 = Ry[j] + cos_value[j] * (r_role + thickness_border)
            x2 = x1 + Rx[j + 1] - Rx[j]
            y2 = y1 + Ry[j + 1] - Ry[j]
            img = cv2.line(img, coordinate_process(x1, y1, offset_x, offset_y),
                           coordinate_process(x2, y2, offset_x, offset_y),
                           color_black, thickness_border, lineType=cv2.LINE_AA)

            x1 = Rx[j] + sin_value[j] * (r_role + thickness_border)
            y1 = Ry[j] - cos_value[j] * (r_role + thickness_border)
            x2 = x1 + Rx[j + 1] - Rx[j]
            y2 = y1 + Ry[j + 1] - Ry[j]
            img = cv2.line(img, coordinate_process(x1, y1, offset_x, offset_y),
                           coordinate_process(x2, y2, offset_x, offset_y),
                           color_black, thickness_border, lineType=cv2.LINE_AA)

        cv2.imwrite('./static/images/previewImgForce/' + str(i) + '.jpg', img)

    return HttpResponseRedirect(reverse('ForceSyn1', args=(num, )))


def PositionSyn(request, num=0, resultIndex=0):
    mechanical_num = nBar_design.num_machanical
    # uniformtiming = nBar_design.timing
    #
    # with open('./static/images/timing.txt', 'w+') as f:
    #     for timing in uniformtiming:
    #         f.write("%.2f\n" % timing)

    return render(request, 'PositionSyn.html', {'num': num,'resultIndex':resultIndex,
                                           'mechanical': [i for i in range(mechanical_num)],
                                           'mechanical_num':mechanical_num})


def ForceSyn(request,num=0,resultIndex=0):
    mechanical_num = force_design.num_machanical
    # uniformtiming = force_design.timing
    #
    # with open('./static/images/timing.txt', 'w+') as f:
    #     for timing in uniformtiming:
    #         f.write("%.2f\n" % timing)

    return render(request, 'ForceSyn.html', {'num': num,'resultIndex':resultIndex,
                                           'mechanical': [i for i in range(mechanical_num)],
                                           'mechanical_num':mechanical_num})


def mechD(resultIndex,function):
    index = int(resultIndex)
    # status_video = int(request.POST['status_video'])

    path = Path('./static/images/result_video_expand/' + str(index))
    if path.exists():
        shutil.rmtree('./static/images/result_video_expand/' + str(index))
    os.mkdir('./static/images/result_video_expand/' + str(index))


    num = function.numPoints
    r_hole = 5
    r_role = 15
    r_wheel = [[25, 30], [35, 40], [48, 58]]
    color_black = (0, 0, 0)
    color_wheel = [(79, 79, 47), (139, 134, 0), (79, 79, 47)]
    color_hole = (255, 255, 255)
    color_track = (0, 255, 127)
    thickness_track = 2
    thickness_border = 1
    color_pole = [(66, 121, 238), (105, 105, 105), (66, 121, 238), (105, 105, 105)]

    R0x = function.R0x[index]
    R0y = function.R0y[index]
    num_role = function.role_num[index]
    phase = function.phase[index]
    length = function.length[index]

    belts = []
    with open('./static/images/result_video_expand/%d/mechtrack.txt' % index, 'w+') as mech:
        for j in range(num_role - 1):
            r1 = r_wheel[num_role - 2 - j][1] - 1
            r2 = r_wheel[num_role - 2 - j][0] - 1
            r = r_wheel[num_role - 2 - j][1] - r_wheel[num_role - 2 - j][0]
            l = length[j]
            delta_x1 = r1 * r / l
            delta_y1 = math.sqrt(r1 ** 2 - delta_x1 ** 2)
            delta_x2 = r2 * r / l
            delta_y2 = math.sqrt(r2 ** 2 - delta_x2 ** 2)

            belt = [[[delta_x1, delta_y1], [l + delta_x2, delta_y2]],
                    [[delta_x1, -delta_y1], [l + delta_x2, -delta_y2]]]
            belts.append(belt)

        track = []
        for i in range(num):
            # 640 x 480 -> 800 x 600
            raw_img = cv2.imread('./static/images/track_video/%d.jpg' % i)
            # img = np.ones([600, 800, 3], np.uint8) * 255
            # img[60:540, 80:720] = raw_img

            img = np.ones([480, 640, 3], np.uint8) * 255
            img[0:480, 0:640] = raw_img

            Rx = []
            Ry = []
            prev_x = R0x
            prev_y = R0y
            cos_value = []
            sin_value = []
            for j in range(num_role):
                cos_ = math.cos(math.radians(phase[j][i]))
                sin_ = math.sin(math.radians(phase[j][i]))
                prev_x += length[j] * cos_
                prev_y += length[j] * sin_
                cos_value.append(cos_)
                sin_value.append(sin_)
                Rx.append(prev_x)
                Ry.append(prev_y)

            Rx.insert(0, R0x)
            Ry.insert(0, R0y)
            Rx_ = Rx[:]
            Ry_ = Ry[:]


            for j in range(num_role + 1):
                Rx_[j], Ry_[j] = coordinate_process(Rx[j], Ry[j])

            # mech.write("%d %d\n" % (Rx_[num_role]-80, Ry_[num_role]-60))
            mech.write("%d %d\n" % (Rx_[num_role]-80, Ry_[num_role]-60))
            track.append((Rx_[num_role]-80, Ry_[num_role]-60))


            for j in range(num_role - 1):
                for belt in belts[j]:
                    x1 = belt[0][0] * cos_value[j] - belt[0][1] * sin_value[j] + Rx[j]
                    y1 = belt[0][0] * sin_value[j] + belt[0][1] * cos_value[j] + Ry[j]
                    x2 = belt[1][0] * cos_value[j] - belt[1][1] * sin_value[j] + Rx[j]
                    y2 = belt[1][0] * sin_value[j] + belt[1][1] * cos_value[j] + Ry[j]
                    img = cv2.line(img, coordinate_process(x1-80, y1+60),
                                   coordinate_process(x2-80, y2+60), color_wheel[j], 2, lineType=cv2.LINE_AA)
                img = cv2.circle(img, (Rx_[j]-80, Ry_[j]-60), r_wheel[num_role - 2 - j][1], color_wheel[j], -1,
                                 lineType=cv2.LINE_AA)
                img = cv2.circle(img, (Rx_[j + 1]-80, Ry_[j + 1]-60), r_wheel[num_role - 2 - j][0], color_wheel[j], -1,
                                 lineType=cv2.LINE_AA)

            for j in range(num_role):

                img = cv2.circle(img, (Rx_[j]-80, Ry_[j]-60), r_role, color_pole[j], -1, lineType=cv2.LINE_AA)
                img = cv2.circle(img, (Rx_[j]-80, Ry_[j]-60), r_role + thickness_border, color_black, thickness_border,
                                 lineType=cv2.LINE_AA)

                if j + 1 == num_role:
                    img = cv2.circle(img, (Rx_[j + 1]-80, Ry_[j + 1]-60), r_role, color_pole[j], -1, lineType=cv2.LINE_AA)
                    img = cv2.circle(img, (Rx_[j + 1]-80, Ry_[j + 1]-60), r_role + thickness_border, color_black, thickness_border,
                                     lineType=cv2.LINE_AA)

                img = cv2.line(img, (Rx_[j]-80, Ry_[j]-60), (Rx_[j + 1]-80, Ry_[j + 1]-60),
                               color_pole[j], 2 * r_role, lineType=cv2.LINE_AA)
                img = cv2.circle(img, (Rx_[j]-80, Ry_[j]-60), r_hole, color_hole, -1, lineType=cv2.LINE_AA)
                img = cv2.circle(img, (Rx_[j]-80, Ry_[j]-60), r_hole + 1, color_black, 1, lineType=cv2.LINE_AA)
                x1 = Rx[j] - sin_value[j] * (r_role + thickness_border)
                y1 = Ry[j] + cos_value[j] * (r_role + thickness_border)
                x2 = x1 + Rx[j + 1] - Rx[j]
                y2 = y1 + Ry[j + 1] - Ry[j]
                img = cv2.line(img, coordinate_process(x1-80, y1+60),
                               coordinate_process(x2-80, y2+60),
                               color_black, thickness_border, lineType=cv2.LINE_AA)

                x1 = Rx[j] + sin_value[j] * (r_role + thickness_border)
                y1 = Ry[j] - cos_value[j] * (r_role + thickness_border)
                x2 = x1 + Rx[j + 1] - Rx[j]
                y2 = y1 + Ry[j + 1] - Ry[j]
                img = cv2.line(img, coordinate_process(x1-80, y1+60),
                               coordinate_process(x2-80, y2+60),
                               color_black, thickness_border, lineType=cv2.LINE_AA)

            for j in range(i - 1):
                img = cv2.line(img, track[j], track[j + 1], color_track, thickness_track)

            cv2.imwrite('./static/images/result_video_expand/%d/%d.jpg' % (index, i), img)


def para(request, num=0, status_video=0, index=-1,resultIndex=0):

    mechD(resultIndex,nBar_design)

    # linknum = 3  # 杆数
    # resultNum = 8  # 总共的机构个数
    # resultIndex = int(request.POST['resultIndex'])  # 选择的机构号码  "linknum": [i for i in range(1,num_pole+1)]
    video_list = []
    mechanical_num = nBar_design.num_machanical  # 生成的机构个数

    hipMotion = nBar_design.hipMotion
    phase_limb = []  # 上臂弧度
    for i in range(len(hipMotion)):
        theta = math.atan((-hipMotion[i,1]+370)/(hipMotion[i,0]-470)) + math.pi/2  #弧度 370/470固定的肘关节位置
        phase_limb.append(theta)

    length = nBar_design.length
    linkLength = length[resultIndex]

    num_pole = nBar_design.role_num  # 机构杆数的[]
    length_list = []
    error = nBar_design.error  # 拟合误差的[]
    linkNum = num_pole[resultIndex]  # 选中的机构的杆数
    linkError = error[resultIndex]

    vxy = nBar_design.endVelocity
    endVelocity = vxy[resultIndex]

    axy = nBar_design.endAcceleration
    endAcceleration = axy[resultIndex]

    initialPhase = nBar_design.phase[resultIndex]  # 初始相位角的[]
    initialPhase_list = []
    velocity = nBar_design.velocity[resultIndex]

    for i in range(linkNum):
        initialPhase_list.append(initialPhase[i][0])

    fixpivot_x = nBar_design.R0x
    fixpivot_y = nBar_design.R0y


    for i in range(mechanical_num):
        tmp = []
        tmp1 = []
        for j in range(num_pole[i]):
            tmp.append(length[i][j])
        length_list.append(tmp)

    for i in range(mechanical_num):
        if status_video >> i & 1 == 1:
            video_list.append(i)

    num_role = nBar_design.role_num[resultIndex]
    phase = nBar_design.phase[resultIndex]
    phase_list = []
    for p in phase:
        tmp1 = []
        for i in range(num):
            tmp1.append(p[i])
        phase_list.append(tmp1[:])

    timing = np.loadtxt('./static/images/timing.txt')
    motion = np.loadtxt('./static/images/result_video_expand/%d/mechtrack.txt' % resultIndex)
    timing_ = []
    motionx = []
    motiony = []
    for i in timing:
        timing_.append(i)

    for i in motion:
        motionx.append(i[0])
        motiony.append(i[1])

    return render(request, 'para.html', {"resultIndex":resultIndex,
                                         'num': num,
                                         "mechanical": [i for i in range(mechanical_num)],
                                         'linkNum': [i for i in range(linkNum)],
                                         'status_video': status_video,
                                         'linkError': linkError,'error_list':error,
                                         'linkLength': linkLength,
                                         'fixpivot_x': fixpivot_x[resultIndex],
                                         'fixpivot_y': fixpivot_y[resultIndex],
                                         'index': index,'initialPhase':initialPhase_list,
                                         'phases':phase_list,
                                         'num_role':num_role,'velocity':velocity,
                                         'timing': timing_,
                                         'motionx': motionx,
                                         'motiony': motiony,
                                         'endVelocity':endVelocity,
                                         'endAcceleration':endAcceleration,
                                         'phase_limb':phase_limb
                                         })


def paraForce(request, num=0, status_video=0, index=-1, resultIndex=0):

    mechD(resultIndex,force_design)
    video_list = []
    mechanical_num = force_design.num_machanical  # 生成的机构个数

    hipMotion = force_design.hipMotion
    phase_limb = []  # 上臂弧度
    for i in range(len(hipMotion)):
        theta = math.atan((-hipMotion[i,1]+370)/(hipMotion[i,0]-470)) + math.pi/2  #弧度 370/470固定的肘关节位置
        phase_limb.append(theta)

    length = force_design.length
    linkLength = length[resultIndex]

    num_pole = force_design.role_num  # 机构杆数的[]
    length_list = []
    error = force_design.error  # 拟合误差的[]
    linkNum = num_pole[resultIndex]  # 选中的机构的杆数
    linkError = error[resultIndex]

    vxy = force_design.endVelocity
    endVelocity = vxy[resultIndex]

    axy = force_design.endAcceleration
    endAcceleration = axy[resultIndex]

    initialPhase = force_design.phase[resultIndex]  # 初始相位角的[]
    initialPhase_list = []
    velocity = force_design.velocity[resultIndex]

    for i in range(linkNum):
        initialPhase_list.append(initialPhase[i][0])

    fixpivot_x = force_design.R0x
    fixpivot_y = force_design.R0y

    for i in range(mechanical_num):
        tmp = []
        tmp1 = []
        for j in range(num_pole[i]):
            tmp.append(length[i][j])
        length_list.append(tmp)

    for i in range(mechanical_num):
        if status_video >> i & 1 == 1:
            video_list.append(i)

    num_role = force_design.role_num[resultIndex]
    phase = force_design.phase[resultIndex]
    phase_list = []
    for p in phase:
        tmp1 = []
        for i in range(num):
            tmp1.append(p[i])
        phase_list.append(tmp1[:])

    timing = np.loadtxt('./static/images/timing.txt')
    motion = np.loadtxt('./static/images/result_video_expand/%d/mechtrack.txt' % resultIndex)
    timing_ = []
    motionx = []
    motiony = []
    for i in timing:
        timing_.append(i)
    for i in motion:
        motionx.append(i[0])
        motiony.append(i[1])

    return render(request, 'paraForce.html', {"resultIndex":resultIndex,
                                              'num': num,
                                              "mechanical": [i for i in range(mechanical_num)],
                                              'linkNum':[i for i in range(linkNum)],
                                              'status_video': status_video,
                                              'linkError':linkError,'error_list':error,
                                              'linkLength':linkLength,
                                              'fixpivot_x':fixpivot_x[resultIndex],'fixpivot_y':fixpivot_y[resultIndex],
                                              'index': index,'initialPhase':initialPhase_list,
                                              'phases':phase_list,
                                              'num_role':num_role,'velocity':velocity,
                                              'timing': timing_,
                                              'motionx': motionx,
                                              'motiony': motiony,
                                              'endVelocity': endVelocity,
                                              'endAcceleration': endAcceleration,
                                              'phase_limb': phase_limb
                                              })


def mechanical_3D(request, resultIndex):
    dict = {}
    num_role = nBar_design.role_num[resultIndex]
    num = nBar_design.numPoints
    phase = nBar_design.phase[resultIndex]
    phase_list = []
    for p in phase:
        tmp = []
        for i in range(num):
            tmp.append(p[i])
        phase_list.append(tmp[:])

    length = nBar_design.length[resultIndex]

    dict['length'] = length
    dict['phases'] = phase_list
    dict['num'] = num
    dict['num_role'] = num_role

    return render(request, '', dict)


def ratioindex(request):
    num = request.POST['num']
    resultIndex = request.POST['resultIndex']
    return HttpResponseRedirect(reverse('para1', args=(num,resultIndex)))


def ratioindexForce(request):
    num = request.POST['num']
    resultIndex = request.POST['resultIndex']
    flag_video = 1
    return HttpResponseRedirect(reverse('paraForce1', args=(num,resultIndex)))

import numpy as np
import cv2 as cv
def scan(img):
    def rowscan(img):
        jump,row,col= np.zeros(img.shape[0]),img.shape[0],img.shape[1]
        for i in range(row):
            for j in range(col-1):
                if img[i,j]!=img[i,j+1]:
                    jump[i]+=1
        return jump
    def isRight(jump):
        begin = False
        row,p_begin,per,Max,k = len(jump),[],-1,0,0
        for i in range(row):
            if jump[i]>=12 and begin:
                continue
            if jump[i]>=12 and not begin:
                p_begin.append([i,row-1])
                begin = True
            if jump[i]<12 and begin:
                p_begin[-1][1],per = i,per+1
                l = i - p_begin[-1][0]
                if l>Max:
                    Max = l
                    k = per
                begin = False
            if jump[i]<12 and not begin:
                continue
        return p_begin[k]
    [begin,end]=isRight(rowscan(img))
    return img[begin:end,:]
                
            
def cut(img1):
    image1, contours, hier = cv.findContours(img1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    point,wid= [],[]
    possible_l = []
    he,wi = img1.shape[0],img1.shape[1]
    for c in contours:
        rect = cv.minAreaRect(c)
        w,h = rect[1][0],rect[1][1]
        box1 = cv.boxPoints(rect)
        box1 =np.int0(box1)
        for i in range(4):
            if box1[i][0]<0:
                box1[i][0] = 0
            if box1[i][0]>wi-1:
                box1[i][0] = wi-1
            if box1[i][1]<0:
                box1[i][1] = 0
            if box1[i][1]>he-1:
                box1[i][1] = he-1
        xs = [i[0] for i  in box1]
        ys = [i[1] for i  in box1]
        x1,x2,y1,y2 = min(xs),max(xs),min(ys),max(ys)
    
        height,width = y2-y1,x2-x1
        if height<0.6*he:
            continue
        #temp[:,:5],temp[:,10:] = 0,0
        #temp[:5,5:10],temp[10:,5:10]=0,0
        temp = img1[y1:y1+height,x1:x1+width].copy()
        possible_l.append(temp)
        point.append(rect[0][0])
        wid.append(width)
    p = point.copy()
    p.sort()

    k,Max = 0,0
    for i in range(len(point)):
        if p[i]>=wi/9 and p[i]<=3*wi/7:
            l = p[i+1] - p[i]
            if l>Max:
                Max ,k= l,i
    k_c = point.index(p[k])
    wid_c = int(wid[k_c])

    char = []
    for i in range(7):
        if i==0:
            k_c = point.index(p[k])
            wid_c = int(wid[k_c])
            pro_left,pro_right = int(p[k]-1.6*wid_c),int(p[k]-0.6*wid_c)
            if pro_left<0:
                pro_left = 0
            char.append(img1[:,pro_left:pro_right])
        else:
            k_c = point.index(p[k+i-1])
            wid_this = wid[k_c]
            if wid_this<0.6*wid_c:
                this_left,this_right = int(p[k+i-1]-0.3*wid_c),int(p[k+i-1]+0.3*wid_c)
                char.append(img1[:,this_left:this_right])
            elif wid_this>1.3*wid_c:
                this_left,this_right = int(p[k+i-1]-0.5*wid_c),int(p[k+i-1]+0.5*wid_c)
                char.append(img1[:,this_left:this_right])
                
            else:
                char.append(possible_l[k_c])
    return char
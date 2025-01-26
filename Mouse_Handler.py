import pyautogui

speed = 1

def move(initial, final):
    if(initial!=final):
        nextX = initial[0]
        nextY = initial[1]
        if(initial[0]>final[0]):
            nextX-=1*speed
        elif(initial[0]<final[0]):
            nextX+=1*speed
        if (initial[1] > final[1]):
            nextY -= 1*speed
        elif (initial[1] < final[1]):
            nextY += 1*speed
        pyautogui.moveTo(nextX,nextY)
        move([nextX,nextY],final)

move([pyautogui.position()[0],pyautogui.position()[1]],[500,500])
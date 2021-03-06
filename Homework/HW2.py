import copy  # 使用深複製 不需pip
state = [0, 0, 0, 0]  # 人、狼、羊、甘藍菜
nextstate = [0, 0, 0, 0]
fullpath = [[0, 0, 0, 0]]

def isdead():  # 需避免條件
    global nextstate
    if nextstate[1] == nextstate[2] and nextstate[2] != nextstate[0]: #狼吃羊
        return True
    elif nextstate[2] == nextstate[3] and nextstate[2] != nextstate[0]: #羊吃菜
        return True
    return False

def checksuccess():  # 檢查是否成功
    global state
    if state[0] == 1 and state[1] == 1 and state[2] == 1 and state[3] == 1:
        return True
    return False

def move():
    global state, nextstate, fullpath
    ship = 1
    nextstate[0] = 0 if state[0] == 1 else 1    #判斷人在此岸還彼岸
    for i in range(1, 4):
        if state[i] == state[0] and ship < 2:  # 跟人同岸且船還沒坐滿兩個
            nextstate[i] = 0 if state[i] == 1 else 1  
            ship += 1
        else:
            continue
        if repeatcheck():  # 如果狀態重複了
            ship = 1
            nextstate[i] = state[i]
        else:
            if isdead():  # 如果發生死亡
                ship = 1
                nextstate[i] = state[i]
            else:
                break
    temp = copy.deepcopy(nextstate)  
    state = temp
    fullpath.append(temp)


def repeatcheck():
    global nextstate, fullpath
    for path in fullpath:  # 檢查有沒有有沒有同樣的狀態
        if nextstate == path:
            return True
    return False

def main():
    global fullpath
    while True:  # 移動直到成功為止
        if checksuccess():
            break
        else:
            move()
    print("path:")
    for path in fullpath:
        print("\t" + str(path))
    print("\nsuccess!")


main()
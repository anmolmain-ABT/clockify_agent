li = [30, 10, 10, 5]
temp=[]
for x in range(0,len(li)-1):
    sub=(li[x+1:len(li)])
    sub.sort()
    print(len(sub),"  :  ",sub[len(sub)-1])
    if li[x] <= sub[len(sub)-1]:
        temp.append((sub[len(sub)-1]))
        
    
print(list(set(temp)))

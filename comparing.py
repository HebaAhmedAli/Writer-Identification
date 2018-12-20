def comparing(path1,path2):
    results=[]
    expectedResults=[]
    with open(path1, "r") as file:
            for line in file:
               results.append(line) 
               
    with open(path2, "r") as file:
            for line in file:
               expectedResults.append(line) 
    correct=0
    for i in range(len(results)):
        if results[i]==expectedResults[i]:
            correct+=1
    print("Performance = "+str(correct/len(results)))
        
                
comparing("results.txt","expected_output.txt")
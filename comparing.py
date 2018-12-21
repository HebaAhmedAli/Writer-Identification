def comparing(path1,path2):
    results=[]
    expectedResults=[]
    with open(path1, "r") as file:
            for line in file:
               results.append(line[0]) 
               
    with open(path2, "r") as file:
            for line in file:
               expectedResults.append(line[0]) 
    correct=0
    for i in range(len(results)):
        if results[i]==expectedResults[i]:
            correct+=1
    print("Performance = "+str(correct/len(results)))
        
def comparingAllOne(path1):
    resultsLen=0.0
    with open(path1, "r") as file:
            correct=0.0
            for line in file:
               resultsLen+=1
               if line[0]=="1":
                   correct+=1
    print("Performance One = "+str(correct/resultsLen))
        
        
comparing("results.txt","expected_output.txt")
comparingAllOne("results.txt")
function countGrade(array){
    let result = 0
    for(let i=0;i<array.length;i++){
        if(array[i] >= 90){
            result += 1
        }
    }
    return result
}

let grads = [90,82,100,70,80]
let result = countGrade(grads)
console.log(result)

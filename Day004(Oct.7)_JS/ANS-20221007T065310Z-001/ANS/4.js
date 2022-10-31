function evenSum(array){
    let result = 0
    for (let i=0;i<array.length;i++){
        if(array[i] % 2 == 0){
            result += array[i]
        }
    }
    return result
}

let numbers = [10,21,30]
let result = evenSum(numbers)
console.log(result)
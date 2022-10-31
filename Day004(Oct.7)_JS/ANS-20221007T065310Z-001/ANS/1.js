function sum(array){
    let result = 0
    for (let i=0;i<array.length;i++){
        result += array[i]
    }
    return result
}

let numbers = [10,20,30]
let result = sum(numbers)
console.log(result)
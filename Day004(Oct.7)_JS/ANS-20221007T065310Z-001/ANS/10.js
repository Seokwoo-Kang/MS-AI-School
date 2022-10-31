function avg(array){
    let result = 0
    let count = 0
    for(let i=0;i<array.length;i++){
        if(array[i] >= 90){
            result += array[i]
            count += 1
        }
    }
    return result / count
}

let grads = [90,82,100,70,80]
let result = avg(grads)
console.log(result)

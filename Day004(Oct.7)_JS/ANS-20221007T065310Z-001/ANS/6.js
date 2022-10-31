function totalSum(n){
    let result = 0
    for(let i = n; i > 0; i--){
        result += i
    }
    return result
}

let num = 11
let result = totalSum(num)
console.log(result)

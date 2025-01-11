//filter: ─
let filterHorizontal: [[Int]] = [
    [1, 1, 1],
    [0, 0, 0],
    [-1, -1, -1]
]
//filter: │
let filterVertical: [[Int]] = [
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
]

//filter: ╱
let filterSlash: [[Int]] = [
    [-1, 1, 2],
    [1, 2, -1],
    [2, -1, -1]
]

//filter: ╲
let filterBackslash: [[Int]] = [
    [2, 1, -1],
    [-1, 2, 1],
    [-1, -1, 2]
]

//filter: ┼
let filterCross: [[Int]] = [
    [-1, 1, -1],
    [1, 4, 1],
    [-1, 1, -1]
]

// //filter: ╳
// let filterX: [[Int]] = [
//     [1, -1, 1],
//     [-1, 4, -1],
//     [1, -1, 1]
// ]

//filter: ├
let filterVerticalRight: [[Int]] = [
    [2, -1, -2],
    [4, 2, 1],
    [2, -1, -2]
]

//filter: ┤
let filterVerticalLeft: [[Int]] = [
    [-2, -1, 2],
    [1, 2, 4],
    [-2, -1, 2]
]

//filter: ┌
let filterLeftTop: [[Int]] = [
    [2, 1, 1],
    [1, -1, -1],
    [1, -1, -2]
]

//filter: ┐
let filterRightTop: [[Int]] = [
    [1, 1, 2],
    [-1, -1, 1],
    [-2, -1, 1]
]

//filter: └
let filterLeftBottom: [[Int]] = [
    [1, -1, -2],
    [1, -1, -1],
    [2, 1, 1]
]

//filter: ┘
let filterRightBottom: [[Int]] = [
    [-2, -1, 1],
    [-1, -1, 1],
    [1, 1, 2]
]

//filter: <
let filterLess: [[Int]] = [
    [-1, 1, -1],
    [2, -1, -2],
    [-1, 1, -1]
]

//filter: ∠
let filterAngle: [[Int]] = [
    [-2, -1, 1],
    [-1, 1, -1],
    [2, 1, 1]
]
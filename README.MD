<link href="misc/style.css" rel="stylesheet"></link>

# AI for SEA - Safety Challenge

![AI for SEA Safety Challenge](misc/Safety.png?style=center)

> Grab has been proactively pushing to make transportation in SEA safer. 
As part of the effort, we want to identify dangerous drivings in a timely manner.

This model will predict dangerous trips based on telematics data.

## How to make prediction?

Run following commands:

```sh
cd main
python <input_path> <output_path>
```

`<input_path>` 
is path to a test file in csv format (single file) 
**OR** 
path to a directory containing test files in csv format (may contains multiple file)  

`<output_path>`
is path for output csv file containing prediction result  

Example:

```sh
cd main
python test.csv result.csv
```
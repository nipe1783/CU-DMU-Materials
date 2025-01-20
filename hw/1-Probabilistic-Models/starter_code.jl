import DMUStudent.HW1

#------------- 
# Problem 4
#-------------

# Here is a functional but incorrect answer for the programming question
function f(a, bs)
    multiplied_vectors = [a * b for b in bs]
    result = reduce((x, y) -> max.(x, y), multiplied_vectors)
    return result
end

# You can can test it yourself with inputs like this
a = [1.0 2.0; 3.0 4.0]
@show a
bs = [[1.0, 2.0], [3.0, 4.0]]
@show bs
@show f(a, bs)

# This is how you create the json file to submit
HW1.evaluate(f, "nipe1783@colorado.edu")

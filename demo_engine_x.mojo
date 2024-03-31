from momograd.x.engine import ValueX

fn main() raises:

    var a = ValueX(-4.0,"a")
    var b = ValueX(2.0,"b")
    
    var c = a + b
    c.label = 'c'

    var d = a * b + b**3
    d.label = 'd'

    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    
    var e = c - d
    e.label = 'e'

    var f = e**2
    f.label = 'f'

    var g = f / 2.0
    g.label = 'g'

    g += 10.0 / f   
   
    g.backward()

    print(g)
    print(a)
    print(b)

    # print the computational graph ...
    g.print_branch()
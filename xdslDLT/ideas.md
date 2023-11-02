Data Layout Trees:

Types:
```
_Type       :== 'dlt.type' '<' '[' _TensorList ']' '>'
_TensorList :== _TensorType (',' _TensorList)?              // Ordered: _TensorType. Each _TensorType must be unique in
                                                            // the _Type.
_TensorType :== '(' '{' _Members? '}' ',' _Dimentions ')'   // Ordering: _Members then _Dimentions.
_Members    :== _Member (',' _Members)?                     // Ordered: _Member values. Ordering: number of _Member then
                                                            // _Member values. Each _Member must be unique in its
                                                            // _TensorType.
_Member     :== _Struct ':' _MemberName                     // Ordering: _Struct values then _MemberName values.
_Dimentions :== (_Dimention '->')* _Primative               // Ordered: _Dimention values. Ordering: number of 
                                                            // _Dimentions then _Dimention values.
_Dimention  :== _DimentionName:_Extent                      // Ordering: _DimentionName values then _Extent values.
_Extent     :== ?                                           // Ordering: ? then _ExtentVariable Value then number.
              | _ExtentVariable
              | number

_Struct, _MemberName, _DimentionName, _ExtentVariable :== alpha_numeric //Ordering: alphabetic.
_Primative  :== xdsl_type
```

Making Structures:
```mlir

builtin.module {

// this specifies the actual physical data layout.
// this can be rewritten as long as the type output constaint checks out.
// preserving the output type guarantees that all accesses are safe.
// the rewrite soundness is dependent on more factors than the types, but
// preserving types assures all accesses still find *some* data...
%tree = dlt.struct {
    %f = dlt.float32 : !dlt.type<[({},f32)]>
    %index = dlt.IndexRange : !dlt.type<[({},idxR)]>

    %A = dlt.dense %index {name = "A", extent = 10} : !dlt.type<[({},A:10->idxR)]>

    %i = dlt.coo %f {name = "I", extent = 5} : !dlt.indexedType<[({},I:5->f32)]>
    %j = dlt.coo %f {name = "J", extent = 5} : !dlt.indexedType<[({},J:5->f32)]>

    %idx1 = dlt.indexing %A into %i : !dlt.type<[({},A:10->I:5->f32)]>

    %idx2 = dlt.indexing %A into %j : !dlt.type<[({},A:10->J:5->f32)]>

    %e = dlt.member <root:e> %idx1 : !dlt.type<[({root:e},A:10->I:5->f32)]>

    %l = dlt.member <root:l> %idx2 : !dlt.type<[({root:l},A:10->J:5->f32)]>

    dlt.yield %l, %e
} : !dlt.type<[({root:e},A:10->I:5->f32), ({root:l},A:10->J:5->f32)]>

// This tree represents the following C-like data layout

struct root {
index* i_coo_indices; // |               |> i pointers           -> Points to a heap buffer of indices as in CSR format
float* i_coo_values;  // |> e, idx1      |                       -> Points to a heap buffer of %f as in CSR format
index[10+1] idx1_A;   // |          |> A                         -> Indexes into the above two buffers as in CSR format

index* j_coo_indices; // |               |> j pointers           -> Points to a heap buffer of indices as in CSR format
float* j_coo_values;  // |> l, idx1      |                       -> Points to a heap buffer of %f as in CSR format
index[10+1] idx2_A;   // |          |> A                         -> Indexes into the above two buffers as in CSR format
}


// full access
dlt.get %tree(root: e, A: %c1, I: %c3) :
    (!dlt.type<[({root:e},A:10->I:5->f32), ({root:l},A:10->J:5->f32)]>, %index, %index) -> f32

// partial access:
dlt.get %tree(root: e, A: %c1) :
    (!dlt.type<[({root:e},A:10->I:5->f32), ({root:l},A:10->J:5->f32)]>, %index) -> !dlt.datatype<[({},I:5->f32)]>

// invalid access,
// there are 2 tensors in %tree:
//                              1: ({root:e},A:10->I:5->f32)
//                              2: ({root:l},A:10->J:5->f32)
// root:e, J:%c3 does not fully appear in either so both are removed from the type leaving us with !dlt.type<[]> which
// is invalid.
dlt.get %tree(root: e, J: %c3) :
    (!dlt.type<[({root:e},A:10->I:5->f32), ({root:l},A:10->J:5->f32)]>, %index) -> !dlt.datatype<[]>

// partial access into struct half, allowed
// let's think about that, shall we
// there are 2 tensors in %tree:
//                              1: ({root:e},A:10->I:5->f32)
//                              2: ({root:l},A:10->J:5->f32)
// J:%c3 exists in 2 but not 1 so 1 is removed: !dlt.type<[({root:l},A:10->J:5->f32)]>
// then the J dimention is 'accessed' leaving us with !dlt.type<[({root:l},A:10->f32)]>
dlt.get %tree(J: %c3) :
    (!dlt.type<[({root:e},A:10->I:5->f32), ({root:l},A:10->J:5->f32)]>, %index) -> !dlt.type<[({root:l},A:10->f32)]>

// Definitely allowed:
dlt.get %tree(A: %c3) :
    (!dlt.type<[({root:e},A:10->I:5->f32), ({root:l},A:10->J:5->f32)]>, %index)
    -> !dlt.type<[({root:e},i:5->f32), ({root:l},j:5->f32)]>

}
```


```
// This struct holds 3 1-Tensors: e,l, and d. They are in the struct of arrays format.
%tree2 = dlt.struct {
    %f = dlt.float32 : !dlt.type<[({},f32)]>

    %A = dlt.dense %f {name = "A", extent = 10} : !dlt.type<[({},A:10->f32)]>
    
    %B = dlt.dense %f {name = "B", extent = 5} : !dlt.type<[({},B:5->f32)]>

    %e = dlt.member <root:e> %A : !dlt.type<[({root:e},A:10->f32)]>
    %l = dlt.member <root:l> %A : !dlt.type<[({root:l},A:10->f32)]>
    %d = dlt.member <root:d> %B : !dlt.type<[({root:d},B:5->f32)]>

    dlt.yield %e, %l, %d
} :  !dlt.type<[({root:d},B:5->f32), ({root:e},A:10->f32), ({root:l},A:10->f32)]>

// In this version we have the same data & same type - but now we put the e and l parts into an array of structs form
%tree3 = dlt.struct {
    %f = dlt.float32 : !dlt.type<[({},f32)]>
    
    %el_struct = dlt.struct {
        %f = dlt.primitive of f32 : !dlt.type<[({},f32)]>
        %e = dlt.member <root:e> %f : !dlt.type<[({root:e},f32)]>
        %l = dlt.member <root:l> %f : !dlt.type<[({root:l},f32)]>
        dlt.yield %e, %l
    } : !dlt.type<[({root:e},f32), ({root:l},f32)]>
    
    %A = dlt.dense %el_struct {name = "A", extent = 10} : !dlt.type<[({root:e},A:10->f32), ({root:l},A:10->f32)]>
    
    %B = dlt.dense %f {name = "B", extent = 5} : !dlt.type<[({},B:5->f32)]>

    %d = dlt.member <root:d> %B : !dlt.type<[({root:d},B:5->f32)]>

    dlt.yield %A, %d
} : !dlt.type<[({root:d},B:5->f32), ({root:e},A:10->f32), ({root:l},A:10->f32)]>


%tree4 = dlt.struct {
    %f = dlt.float32 : !dlt.type<[({},f32)]>
    
    %node = dlt.struct {
        %f = dlt.primitive : !dlt.type<[({},f32)]>
        %e = dlt.member <node:e> %f : !dlt.type<[({node:e},f32)]>
        %l = dlt.member <node:l> %f : !dlt.type<[({node:e},f32)]>
        dlt.yield %e, %l
    } : !dlt.type<[({node:e},f32), ({node:l},f32)]>
    
    %A = dlt.dense %node {name = "A", extent = 10} : !dlt.type<[({node:e},A:10->f32), ({node:l},A:10->f32)]>
    
    %B = dlt.dense %f {name = "B", extent = 5} : !dlt.type<[({},B:5->f32)]>

    %n = dlt.member <root:n> %A : !dlt.type<[({node:e, root:n},A:10->f32), ({node:l, root:n},A:10->f32)]>
    %d = dlt.member <root:d> %B : !dlt.type<[({root:d},B:5->f32)]>

    dlt.yield %n, %d
} : !dlt.type<[({node:e, root:n},A:10->f32), ({node:l, root:n},A:10->f32), ({root:d},B:5->f32)]>


// even though there's only one possible f32 given node:e A:%c1 (as we can see 'root' must be 'n') it still needs to be
// specified 
dlt.get %tree4 (node:e, A: %c1) :
    (!dlt.type<[({node:e, root:n},A:10->f32), ({node:l, root:n},A:10->f32), ({root:d},B:5->f32)]>, %index, %index)
     ->!dlt.type<[({root:n}, f32})]>

dlt.type<[({node:a, root:n}, f32),({node:b, root:n}, f32),({root:c}, f32)]>


// In %tree5 and %tree6 we have the same data to be stored & same type. but in %tree6 the 'root:a' array over A is split
// into two parts I and J which are placed before and after the 'root:b' member in the struct. This is done using index
// aritmetic/Affine to map the A dimention onto the I and J dimentions respectively.
%tree5 = dtl.struct {
    %f = dlt.float32 : !dlt.type<[({},f32)]>
    
    %A = dlt.dense %f {name="A", extent=10} : !dlt.type<[({},A:10->f32)]>
    
    %a dlt.member <root:a> %A : !dlt.type<[({root:a},A:10->f32)]>
    %b dlt.member <root:b> %f : !dlt.type<[({root:b},f32)]>
    
    dlt.yield %a, %b
} : !dlt.type<[({root:a},A:10->f32), ({root:b},f32)]>


%subtree6 = dtl.struct {
    %f = dlt.float32 : f32
    
    %I = dlt.dense %f {name="I", extent=5} : !dlt.type<[({},I:10->f32)]>
    %J = dlt.dense %f {name="J", extent=5} : !dlt.type<[({},J:10->f32)]>
    
    %i dlt.member <root:a, aff:lhs> %I : !dlt.type<[({root:a},I:5->f32)]>
    %j dlt.member <root:a, aff:rhs> %J : !dlt.type<[({root:a},J:5->f32)]>
    %b dlt.member <root:b> %f : !dlt.type<[({root:b},f32)]>
    
    dlt.yield %i, %b, %j
} : !dlt.type<[({root:a, aff:lhs},I:5->f32), ({root:a, aff:rhs},J:5->f32), ({root:b},f32)]>

%tree6 = dlt.indexAffine from [A:10] with affine_set< (a)[]: (a <= 4) > defining {aff:lhs} and affine_map<(a)[]->(a)> to [I:5] else {aff:rhs} and affine_map<(a)[]->(a-5)> to [J:5] on %subtree
    : !dlt.type<[({root:a, aff:lhs},I:5->f32), ({root:a, aff:rhs},J:5->f32), ({root:b},f32)]> -> !dlt.type<[({root:a},A:10->f32), ({root:b},f32)]>
// how to type affine:
// 1. look for the members being defined: aff:lhs and aff:rhs to remove: [({root:a, aff:lhs},I:5->f32), ({root:a, aff:rhs},J:5->f32)]
// 2. replace their Dimentions respectivly [I:5]->[A:10] and [J:5]->[A:10] to get [({root:a, aff:lhs},A:10->f32), ({root:a, aff:rhs},A:10->f32)]
// 3. remove the members respectivly: [({root:a},A:10->f32), ({root:a},A:10->f32)]
// 4. remove duplicates - these should only exist because of the affine_set selection so logically are disjoint sections of the same space: [({root:a},A:10->f32)]
// 5. add these back to the original, replacing the tensors found in step 1: [({root:a},A:10->f32), ({root:b},f32)]



// dlt.indexAffine :== 'dlt.indexAffine' 'from' '[' <Dimentions> (',' <Dimentions>)* ']' 'with' <#set> 'defining <value | {<Members>} and #map to [I:5]> else <value | {<Members>} and #map to [J:5]> on %otherDlt


%tree6 = dlt.indexAffine on %subtree6 from [A:10] with #set defining [<Member|value>, <Member|value>] and #map to [J:5, K:5]
    : !dlt.type<[({root:a, aff:true},I:5->f32), ({root:a, aff:false},J:5->f32), ({root:b},f32)]>
    -> !dlt.type<[({root:a},A:10->f32), ({root:b},f32)]>
    
    
// extent from block arg, diaganl 
%ext = dlt.extent %blockArg as "ex1": !dlt.extentType
%f = dlt.primative of float32 : !dlt.type<[({},f32)]>
%mat = dlt.dense %f {name="A", extent = %ext} : !dlt.type<[({},A:ex1->f32)]>
%diagMat = dlt.member <aff:true> %mat : !dlt.type<[({aff:true},A:ex1->f32)]>
%diag = dlt.indexAffine from [I:ex1, J:ex1] with affine_set<(d0, d1)[]: (d0 =! d1)> defining [aff:true, 0] and affine_map<(d0, d1)[] -> (d1)> to [A:ex1] on %mat
    : !dlt.type<[({},A:ex1->f32)]>
    -> !dlt.type<[({root:a},A:10->f32), ({root:b},f32)]>
    
    
%tree6 = dlt.indexAffine from [A:10] with #set defining <value | {Members} and #map to [I:5]> else <value | {Members} and #map to [J:5]> on %subtree6
    : !dlt.type<[({root:a, aff:true},I:5->f32), ({root:a, aff:false},J:5->f32), ({root:b},f32)]>
    -> !dlt.type<[({root:a},A:10->f32), ({root:b},f32)]>

```

dtl nodes/ops:
```mlir
dlt.struct {
...
dlt.yield %member1, %member2,... 
}
dlt.indexing %1 into %2
dlt.member <Members> %1

dlt.primative of !<type>
dlt.indexRange
dlt.index


dlt.dense
dlt.upcoo

dlt.indexAffine from [<Dimentions>] with #set defining {<Members>} and #map to [<Dimentions>] else {<Members>} and #map to [<Dimentions>] on %1 

```
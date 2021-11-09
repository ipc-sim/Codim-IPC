# JGSL Storage

In this document we do a tutorial of how to use JGSL's Storage works. The `STORAGE` is a wrapper
of Cabana's AoSoA data structure. It hides some implementation detail, provides a clean user API,
but also inherits some technical difficulty from Cabana. You may encounter lots of problems when
using `STORAGE`. So please refer to this document and hope this can solve some of your questions.

## 1. Type Of Storage

In a nutshell, the type of the storage should be looking something like this:

``` C++
STORAGE<Kokkos::Serial, Kokkos::HostSpace, 4, int, VECTOR<float, 3>, VECTOR<float, 3>>
```

If we decompose the template types, we have roughly 4 things to consider:

1. `Kokkos::Serial` - We call this `EX_SPACE`, or **Execution Space**
2. `Kokkos::HostSpace` - We call this `MEM_SPACE`, or **Memory Space**
3. `4` - We call this `BIN_SIZE`, or **Bin Size**
4. `int, VECTOR<float, 3>, VECTOR<float, 3>, ...` - We call these types **Member Types**

In the following sections we are going to discuss how to configure these types more concretely.

### 1.1. Memory & Execution Spaces

Since our Storage uses Kokkos and automatically adapt to multiple memory and execution spaces, we
have them configurable on the template level. Here are a table of potential spaces we can use on
the `STORAGE` class:

|                     | Serial | OpenMP | Threads | Cuda | ROCm |
|---------------------|--------|--------|---------|------|------|
| HostSpace           | x      | x      | x       | -    | -    |
| HBWSpace            | x      | x      | x       | -    | -    |
| CudaSpace           | -      | -      | -       | x    | -    |
| CudaUVMSpace        | x      | x      | x       | x    | -    |
| CudaHostPinnedSpace | x      | x      | x       | x    | -    |
| ROCmSpace           | -      | -      | -       | -    | x    |
| ROCmHostPinnedSpace | x      | x      | x       | -    | x    |

(Reference: https://github.com/kokkos/kokkos/wiki/View#641-memory-spaces)

According to the above table, you can have

``` C++
STORAGE<Serial, HostSpace, ...>
STORAGE<OpenMP, HostSpace, ...>
STORAGE<OpenMP, HBWSpace, ...>
STORAGE<Cuda, CudaSpace, ...>
```

But not

``` C++
STORAGE<OpenMP, CudaSpace, ...> // This is WRONG
```

(apparently you cannot have OpenMP executed on Cuda)

Defining Memory & Execution Spaces correctly will give you ability to directly port this data
structure between CPU/OpenMP and GPU/Cuda, which is very handy.

### 1.2. Bin Size and Memory Alignment

We are using AoSoA as our core data structure of `STORAGE`. This is allowing us for a variety of
benifits. For example, this `AoSoA` structure can be simplified into a simple `A(rray)` or `AoS`
if those are desired.

For example, in terms of memory structure, a

``` C++
STORAGE<..., VECTOR<float, 3>>
```

is simply a

``` C++
std::vector<VECTOR<float, 3>>
```

They both represent **Array** of `VECTOR<float, 3>`.

Another example would be

``` C++
STORAGE<..., 1, VECTOR<float, 3>, VECTOR<float, 3>, int, int, float, float>
```

being equal to the memory structure of

``` C++
std::vector<std::tuple<VECTOR<float, 3>, VECTOR<float, 3>, int, int, float, float>
```

which is just `AoS` (Array Of Struct). Note that `BIN_SIZE` is set to `1` in this case.

Enough of examples. More formally speaking, `AoSoA` is `Array of Structure of Array`. There are
some terms that we need to be super clear of:

- Component Types: an ordered list of types being stored inside this data structure
- Element: a tuple of all the component types
- Component at index: a piece of data of the type in the Component Types at a given index
- Bin: a fix sized structure of array containing fix amount of elements
- #Elem/Bin: the number of elements inside a single bin. This number is fixed and is known as
  `BIN_SIZE`
- #Bin: the number of bins in this AoSoA
- #Elem = #Bin * #Elem/Bin

A visual representation of AoSoA is illustrated here:

![AoSoA](https://github.com/ECP-copa/Cabana/wiki/images/aosoa_layout.png)

(Reference: https://github.com/ECP-copa/Cabana/wiki/AoSoA#overview)

In the above example, the `BIN_SIZE`, or #Elem/Bin, is `4`. Since 4 `x`s are aligned consecuti
vely, and then 4 `y`s, and finally 4 `z`s.

The number `BIN_SIZE` has to be an exponent of `2`. So `2`, `4`, `8`, `16`, and etc., are all
acceptable.

#### Important Note on Memory Alignment

Another minor but very important thing in designing your `STORAGE` is that, the aligned memory
must be castable to a pointer of any member component types. This is a not so novel limitation
from the implementation of Cabana. Unless we implement our AoSoA ourselves, we cannot get
around this limitation. So be prepared to design your data that satisfy the following rule:

For every member type,

```
sizeof(Bin) % sizeof(MemberType) == 0
```

Note that

```
sizeof(Bin) = BIN_SIZE * sizeof(Element)
sizeof(Element) = sum of sizeof(MemberType)
```

As an example,

```
STORAGE<4, VECTOR<float, 3>, float>
```

has a good alignment. The size of `VECTOR<float, 3>` is `4` floats (16 bytes), the size of
float is 4 bytes. Adding them together, we have `sizeof(Element) = 20 bytes`. Note that we
have our `BIN_SIZE = 4`, so `sizeof(Bin) = 4 * 20 = 80 bytes`. And that is dividable by 16,
the size of `VECTOR<float, 3>`, then inherently dividable by 4, which is the size of a
float.

The intrinsic here is that a single `BIN` can be casted to `VECTOR<float, 3>[5]`, where the
first 4 `VECTOR<float, 3>` stored consecutively, and the last "`VECTOR<float, 3>`" is
actually 4 floats. Cabana use this kind of type conversion to access the corresponding data
using slice and strides.

Note that if we set the `BIN_SIZE` to `2`, then the memory cannot align, because then
`sizeof(Bin) = 2 * 20 = 40` which is not dividable by `16`. `BIN_SIZE` of `8` will work
because `sizeof(Bin) = 8 * 20 = 160` and is dividable by `16`. Any exponent of `2` that
is greater than `4` will be good actually.

From this we know, every `STORAGE` containing only a single member type is by default
memory alignable.

But the following example will not work that well:

```
STORAGE<???, MATRIX<float, 3>, float>
```

Let's ignore the `BIN_SIZE` for now.

We have `sizeof(MATRIX<float, 3>) = 48 (12 floats)` and `sizeof(float) = 4`. Adding them
together will get us `sizeof(Element) = 52`. Let's suppose `BIN_SIZE = n`, then what we
want is `(n * 52) % 48 = 0`. But this is never possible when `n` is an exponent of `2`.

In case you wonder how do we solve this problem, we can add two more floats to this:

```
STORAGE<4, MATRIX<float, 3>, float, float, float>
```

Here `sizeof(Element) = 48 + 4 + 4 + 4 = 60`, and `sizeof(Bin) = 4 * 60 = 240`, which is
apparently dividable by `48` itself!

### 1.3. Member Data Types

It's also very crucial to store the correct Member Data Types. In a sentence, we want to
store the **stupid simple** types -- types without virtual functions, no pointer to other
data structures, have default constructor and can be copied easily. You definitely don't
want to store a `STORAGE` inside another `STORAGE`. That is crucial. Ideally, you want
`STORAGE`s to be parallel to each other.

In our code base, all of the following common ones are good to be stored in `STORAGE`:

``` C++
// Scalars
float // 4 bytes
double // 8 bytes

// Vectors
VECTOR<float, 2> // 16 bytes
VECTOR<float, 3> // 16 bytes
VECTOR<float, 4> // 16 bytes
VECTOR<double, 2> // 32 bytes
VECTOR<double, 3> // 32 bytes
VECTOR<double, 4> // 32 bytes

// Matrices
MATRIX<float, 2> // 32 bytes
MATRIX<float, 3> // 48 bytes
MATRIX<float, 4> // 64 bytes
MATRIX<double, 2> // 64 bytes
MATRIX<double, 3> // 96 bytes
MATRIX<double, 4> // 128 bytes
```

I've marked all the data sizes here for memory alignment calculation. In our experience,
`MATRIX<?, 3>` is the hardest to deal with since the data size is a multiple of `3`. That
requires you to make the sum of other things to be a multiple of `3` to secure the alignment.
This could be tricky.

## 2. In Storage Manipulation

### 2.1. Initialization

We should first create type alias for the code to be easier to read

``` C++
using MPM_PARTICLES = STORAGE<
  Kokkos::Serial, Kokkos::HostSpace, 4, // Configuration
  VECTOR<double, 2>, // X
  VECTOR<double, 2>, // V
  MATRIX<double, 2>, // grad_v
  double, // m
  double // padding data
>;
```

And then a storage can be initialized as simple as

``` C++
MPM_PARTICLES particles;
```

Our storage contains a notion of `capacity`. The capacity is by default `100` and if filled
full and want to insert more, the capacity can increase. But the bad thing there is that we
need to reallocate the data with a larger capacity, which can be a slow operation. So it is
your best interest to allocate as much as you need in the first hand. To specify the number
of elements you want to store in this `STORAGE`, you pass a number to the constructor:

``` C++
MPM_PARTICLES particles(500);
```

In this case, the initial capacity is `500` as opposed to the default `100`.

### 2.2. Insertion

To maintain a relationship between storages, we assume that there exists a **global index**
for each of the storage elements. Therefore we require insertion with `index`:

``` C++
for (int i = 0; i < 100; i++) {
  particles.Insert(i, data_1, data_2, data_3, data_4, data_5);
  //               ^ <- this is the index
}
```

If you insert to an index already containing an element, that will be just an `Update` to
the data on that index.

If there are elements in different `STORAGE`s that you want to associate them together, you
certainly want to insert with **the same index**.

``` C++
store_1.Insert(index, data_1, data_2, ...);
store_2.Insert(index, data_a, data_b, ...);
//             ^^^^^ <- Notice the same index
```

It is definitely possible that some element exists in `store_2` and some doesn't. In that
case, you can write something like

``` C++
for (...) {
  store_1.Insert(index, data_1, data_2, ...);
  if (SOME_CRITERIA) {
    store_2.Insert(index, data_a, data_b, ...);
  }
}
```

This will come in handy when you want to `Join` two storages together. But we will talk about
`Join` later.

To keep a unified scheme across storages, you should use our `ID_ALLOCATOR` to give you ID
without you thinking about the alignments.

### 2.3. ID Allocator

An ID Allocator is commonly used when you have a group of inter operating `STORAGE`s.

``` C++
ID_ALLOCATOR allocator;
for (int i = 0; i < 100; i++) {
  auto index = allocator.Alloc();
  //   ^^^^^ <- The allocated index
  store_1.Insert(index, data_1, data_2, ...);
  store_2.Insert(index, data_a, data_b, ...);
}
```

You certainly want to have more ID Allocator if you have multiple groups of inter-operating
`STORAGE`s.

An example of that would be `Mesh`. You have `nodes` and `elems`. Each have some different
types of attribute associated with them. In this case, you want to have a data structure
looking like this:

``` C++
STORAGE<..., VECTOR<T, dim>> nodes;
STORAGE<..., NODE_ATTR_1, NODE_ATTR_2, ...> node_attrs;

STORAGE<..., VECTOR<int, 4>> elems; // 4 node per element (tetrahedron/quad)
STORAGE<..., ELEM_ATTR_1, ELEM_ATTR_2, ...> elem_attrs;
```

Of course `nodes` and `node_attrs` are associated and should contain associated data. It goes
the same for `elems` and `elem_attrs`. We then need to associate two ID Allocators for these
two groups:

``` C++
STORAGE<..., VECTOR<T, dim>> nodes;
STORAGE<..., NODE_ATTR_1, NODE_ATTR_2, ...> node_attrs;
ID_ALLOCATOR node_id_allocator;

STORAGE<..., VECTOR<int, 4>> elems; // 4 node per element (tetra/quad)
STORAGE<..., ELEM_ATTR_1, ELEM_ATTR_2, ...> elem_attrs;
ID_ALLOCATOR elem_id_allocator;
```

### 2.4. Remove

You can remove an element using its **global index**.

``` C++
store.Remove(index);
```

This function will return `true` when removed successfully. When unsuccessful, it's usually
the element being not presented in the storage.

If an element is removed from all of its correspondence in storages, you should consider
letting `ID_ALLOCATOR` to recycle it's global index. In that case, when a new element is
inserted, it can reuse the old global index that some element left empty. To do that, you can
write

``` C++
store_1.Remove(index);
store_2.Remove(index);
// ... Remove all the correspondence in all storages.
allocator.Remove(index);
// Free the index from ID allocator

// ... Some time later...
auto index = allocator.Alloc(); // This index will be the freed up one before
```

Please always free up the index if you know that index could be emptied.

### 2.5. Update

Update behaves very similar to `Insert`. The main difference is that `Update` will return a
boolean value denoting whether the update succeeded or not. If a global index is not presented
in the storage, it will fail - `Update` only updates the elements that are already existed in
the storage.

``` C++
for (int i = 0; i < 50; i++) {
  store.Insert(i, data_1, data_2, ...);
}

store.Update(30, data_1, data_2, ...); // true (Success)
store.Update(100, data_1, data_2, ...); // false (Fail...) because index 100 is not in the store
```

### 2.6. Size

To get the amount of elements stored in the `STORAGE`, do

``` C++
store.size
```

Please **DON'T** write something like this:

``` C++
for (int i = 0; i < store.size; i++) {
  auto data = store.Get(i);
}
```

Because the index between `0` and `store.size` is not necessarily the **global index** of each
element.

### 2.7. Contains Check

To check if an index is contained in the `STORAGE`, do

``` C++
store.Contains(index)
```

### 2.8. Get and Get_Unchecked

You can, of course, get element by index in the `STORAGE`. But there are three variations: `Get`, `Get_Unchecked` and `Get_Unchecked_Const`.

`Get` returns an `std::optional` type of the tuple. It will be your best interest to check if
that element exists or not:

``` C++
auto maybe_data = store.Get(index);
if (maybe_data.has_value()) {
  // maybe_data.value is the actual data contained
  do_something(maybe_data.value());
}
```

`Get_Unchecked` will return you directly the tuple information, but may suffer from possibility
of failing:

``` C++
for (int i = 0; i < 50; i++) {
  store.Insert(i, data_1, data_2, ...);
}

auto data = store.Get_Unchecked(30); // Data at index 30

auto data = store.Get_Unchecked(100); // bad_option_value thrown
```

If you have clear idea of that index being existed in the `STORAGE`, then feel free to use
`Get_Unchecked`. Otherwise, use `Get`.

The result will be returned in the form of tuple. You have several ways to use it. Suppose
we have a `STORAGE` storing `float`, `int`, and `double`:

``` C++
using MY_STORE = STORAGE<..., float, int, double>;

MY_STORE store;

// Get with structured binding
auto maybe_data = store.Get(index);
if (maybe_data.has_value()) {
  auto &[f, i, d] = maybe_data.value();
  // f is the float,
  // i is the int,
  // d is the double
}

// Get with `std::get<Index>`
auto maybe_data = store.Get(index);
if (maybe_data.has_value()) {
  auto &data = maybe_data.value();
  float f = std::get<0>(data);
  float i = std::get<1>(data);
  float d = std::get<2>(data);
}

// Get_Unchecked with structured binding
auto &[f, i, d] = store.Get_Unchecked(index);
```

`Get_Unchecked_Const` is similar to `Get_Unchecked`. But the returned value is stated
`const` and cannot be mutated. This can be used when you want to get something but not
wanting to change it. It will become handy when we are talking about parallel computation
without racing condition.

``` C++
const auto &[f, i, d] = store.Get_Unchecked_Const(index);
```

### 2.9. Iterate with `Each` and `Par_Each`

You always need to iterate through all elements of a single `STORAGE`. In that case, the
function you want to use is `Each` and `Par_Each`. They both possess the same sort function
signatures, but the intrinsics are different. `Each` is serialized each which does not have
requirements on your iterator function. `Par_Each` stands for "Parallel Each" and will assume
that you want to execute the loop body (kernel) in parallel. That will require you to write
functions in a manner that there's no racing conditions.

Our `Each` takes in a "Lambda Function" with no constraint. The function will be provided
a **global index** `i`, and the `data` in the tuple form at that **global index**.

``` C++
using MY_STORE = STORAGE<..., float, int, double>;
MY_STORE store;
store.Each([](int id, auto data) {
  auto &[f, i, d] = data;
  // Do things with `f`, `i` and `d`.
});
```

You can of course do some mutation of closure data

``` C++
int count = 0;
store.Each([&](int, auto) {
  count += 1;
});
assert(count == store.size);
```

In this case, `count` will result in the size of the `STORAGE`. Notice the lambda capture
here is set to `[&]`, which means "capture all the reference of the environment". So `count`
is captured by reference and can be mutated. Since `Each` executes your kernel in a serial
manner, there will be no racing condition being introduced.

For `Par_Each`, there are more constraints. You will need to use `KOKKOS_LAMBDA` here since
we are using `Kokkos` to parallelize the execution.

``` C++
store.Par_Each(KOKKOS_LAMBDA(int i, auto data) {
  auto &[f, i, d] = data;
  // Do things with `f`, `i` and `d`.
});
```

The following example **WILL NOT** work, because of the `KOKKOS_LAMBDA` restriction.

``` C++
int count = 0;
store.Par_Each(KOKKOS_LAMBDA(int, auto) {
  count += 1; // COMPILATION FAILURE!!!!
});
```

The compile time error is

```
  cannot assign to a variable captured by copy in a non-mutable lambda
    count += 1; // COMPILATION FAILURE!!!!
    ~~~~~ ^
```

If we are to solve this problem, we can use Kokkos's `View` and atomic operations:

``` C++
Kokkos::View<int> count("Count");
store.Par_Each(KOKKOS_LAMBDA(int, auto) {
  Kokkos::atomic_add(&count(), 1); // Success!
});
```

We will introduce more on atomic operations in the next section.

The following **WILL** work, since we used `Get_Unchecked_Const` instead of `Get_Unchecked`:

``` C++
using NODES = BASE_STORAGE<VECTOR<float, 3>>;
using ELEMS = BASE_STORAGE<VECTOR<int, 4>>; // Tetras
NODES nodes;
ELEMS elems;
elems.Par_Each(KOKKOS_LAMBDA(int, auto data) {
  auto tetra = std::get<0>(data);
  const auto &[node_1] = nodes.Get_Unchecked_Const(tetra.x);
  const auto &[node_2] = nodes.Get_Unchecked_Const(tetra.y);
  const auto &[node_3] = nodes.Get_Unchecked_Const(tetra.z);
  const auto &[node_4] = nodes.Get_Unchecked_Const(tetra.w);
  // Do things to node_1, 2, 3, and 4
});
```

`Const` here means that the `nodes` storage and each `node_x` itself will not be mutated.
Therefore safe to access during parallel computation.

### 2.10. Atomic in `Par_Each`

> TODO: Add more atomic operation tutorial

### 2.11. Deep Copy

Deep copy can happen between two storages sharing the same member types. The execution &
memory space & bin size do not need to be the same.

``` C++
STORAGE<Serial, HostSpace, 16, int, int, float> local_store;
// Assume local_store is popularized

// Initialize a store on cuda
STORAGE<Cuda, CudaSpace, 4, int, int, float> cuda_store;

// Copy from local store to cuda store
local_store.Deep_Copy_To(cuda_store);
```

## 3. Interop of Storages and Joined Storages

It will be super common to connect the `STORAGE`s that have intersections. If we know that
two or more storages have correspondence, we can simply do

``` C++
store_1.Join(store_2, store_3, ...)
```

We call the resulting store a `JOINED_STORAGE`. Conceptually, you can think of the joined
storage only contains element that presents in all of its member `STORAGE`s. Technically,
joined storage just stores references.

### 3.1. Order of Joined Storages

You might notice that we might have different orders joining storages

``` C++
store_2.Join(store_1, store_3, ...)
```

and that **DOES** matter. Since we use the information in the first `STORAGE` to guide
the iteration, it will be better if we put the `STORAGE` with the **LEAST** amount of elements
at the first place.

Consider the following example:

``` C++
Particles particles;
Deformations deformations;

for (int i = 0; i < 100; i++) {
  particles.Insert(i, ...);
  if (i < 50) {
    deformations.Insert(i, ...);
  }
}
```

It is apparent that `particles` contains `100` elements and `deformations` contains `50`. When
we want to join them together we have two choices. But the later will be better:

``` C++
particles.Join(deformations).Each(...) // Will loop through 100 elements, where the later 50 is
                                       // wasted

deformations.Join(particles).Each(...) // Will loop through 50 elements (of deformations), which
                                       // does not waste any parallel computing power
```

### 3.2. `Each` and `Par_Each` of Joined Storage

Similar to normal `STORAGE`, you can definitely loop through elements of a `JOINED_STORAGE`.
The iteration kernel will be called only when that element is presented in **ALL** `STORAGE`s
in the `JOINED_STORAGE`.

The lambda of both `Each` and `Par_Each` should take the data containing concatenation of all
data of each member storages in order. Take the following example

``` C++
using AB = STORAGE<..., A, B>;
using CD = STORAGE<..., C, D>;

AB ab;
CD cd;

ab.Join(cd).Each([](int, std::tuple<A&, B&, C&, D&> data) {
  auto &[a, b, c, d] = data;
  // ...
});

cd.Join(ab).Each([](int, std::tuple<C&, D&, A&, B&> data) {
  auto &[c, d, a, b] = data;
});
```

The `Each`, `Par_Each`, `Each_Const` functions are all similar to the functions in base
storage.

## 4. Field Labeling

You can label the fields of storages by using `FIELDS_WITH_OFFSET` type trait. It can become
really handy in lots of advanced use cases.

### 4.1. Label a Storage

When you have a storage that you want labels associated, please implement the
`FIELDS_WITH_OFFSET` type trait for that storage. Consider the following example:

``` C++
using POS_VEL_MASS = STORAGE<..., VECTOR3f, VECTOR3f, float>;
```

Say the first component is `position`, second component is `velocity`, and the third
component is `mass`. We can label this storage by

``` C++
template<std::size_t OFFSET>
struct FIELDS_WITH_OFFSET<OFFSET, POS_VEL_MASS> {
  enum INDICES { POSITION = OFFSET, VELOCITY, MASS };
};
```

There are few things to note here:

1. There must be a template variable `OFFSET`, and that should be the first template
   argument to the type trait `FIELDS_WITH_OFFSET`.
2. The second template argument must be the `STORAGE` that you want to label.
3. There must be an `enum` called `INDICES` inside the struct.
4. The fields should be named non-overlapping, and the order should be consistent with
   the storage member types.
5. The first field must be set to equal to the `OFFSET` template variable by
   `FIRST_FIELD = OFFSET`.

Also note that, there are some cases where we put `padding` fields at **the end** of
the storage. You **don't** have to label them. In the following example, we only store
`F` as a matrix, `mu` as a float, `lambda` as a float. But the memory will not align.
Therefore we append another dummy `float` at the end. When we are labeling this storage,
clearly we only need to label the first three elements, and that's indeed what we do!
No need to add a dummy name for a dummy field.

``` C++
using F_MU_LAMBDA = STORAGE<..., MATRIX3f, float, float, float>

template<std::size_t OFFSET>
struct FIELDS_WITH_OFFSET<OFFSET, F_MU_LAMBDA> {
  enum INDICES { F = OFFSET, MU, LAMBDA };
};
```

### 4.2. Use the Field Labels of a Storage

Following the above example, we can use the fields with `FIELDS` type trait

``` C++
POS_VEL_MASS particles;
particles.Each([](int, auto data) {
  VECTOR3f &position = std::get<FIELDS<POS_VEL_MASS>::POSITION>(data);
  //                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ <- Look at this

  VECTOR3f &velocity = std::get<FIELDS<POS_VEL_MASS>::VELOCITY>(data);
  float &mass = std::get<FIELDS<POS_VEL_MASS>::MASS>(data);
});
```

This `FIELDS<POS_VEL_MASS>::POSITION` basically serves as an index. Of course `POSITION`
will be `0` in this case and you will of course get the `0`th element of `data`.
Similarly, you can use `MASS` and `VELOCITY` to access as well.

As an interesting use case, this can provide a uniform programming model when you
have template over storages. Suppose you now have `POS_VEL_MASS` and also `VEL_POS`.
They have different type informations but they both have the field `POSITION`. Now if
you have a function that you want to extract the position from these two storages,
you can do

``` C++
template<typename STORAGE>
VECTOR3f get_position(STORAGE &storage, std::size_t i) {
  auto data = storage.Get_Unchecked(i);
  return std::get<FIELDS<STORAGE>::POSITION>(data);
}

int main() {
  POS_VEL_MASS pos_vel_mass;
  VEL_POS vel_pos;
  auto pos_in_pos_vel_mass = get_position(pos_vel_mass, 10);
  auto pos_in_vel_pos = get_position(vel_pos, 5);
}
```

### 4.3. Labels of a Joined Storage

Given that you have field labels for individual storages, you definitely want to have
field labels for any `JOINED_STORAGE` as well. And fortunately, we do offer that
functionality! To use labels of a Joined Storage, you can do the following

``` C++
template<typename MODEL>
void operation_on_f(PARTICLES &pars, MODEL &model) {
  using JOINED = decltype(pars.Join(model));
  //             ^^^^^^^^^^^^^^^^^^^^^^^^^^ 1. Declare a type of the joined storage

  pars.Join(model).Each([](int, auto data) {
    auto &f = std::get<FIELDS<JOINED>::F>(data);
    //                 ^^^^^^^^^^^^^^^^^ 2. Access that field in the joined fields
  });
}
```

An important note is that you cannot access the joined fields when member storages
share same field names. There will be a compile error if this happen. You can still
`Join` the storages for sure, but you cannot use `FIELDS<JOINED>`. Here's the example

``` C++
using AB = STORAGE<..., A, B>; // Assume AB is labeled `A` and `B`
using BC = STORAGE<..., B, C>; // Assume BC is labeled `B` and `C`
using JOINED = decltype(declval<AB>().Join(declval<BC>())); // Create joined AB/BC,
                                                            // which is fine
FIELDS<JOINED>::B // Compile Error!
```

### Appendix: Todo List

- [ ] Deep copy of the whole data structure
- [x] Documentation of Fields Feature
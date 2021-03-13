#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <iostream>
#include <random>

typedef long long ll;
namespace py = pybind11;

py::array_t<ll> cm_sketch_preds(int nhashes, py::array_t<ll> np_input, ll width, int seed)
{
    //returns the predictions from a count-min sketch
    std::mt19937 gen(seed);

    py::buffer_info buf1 = np_input.request();

    srand(seed);
    ll *ptr1 = static_cast<ll *>(buf1.ptr);

    std::vector<ll> arr;
    for (size_t idx = 0; idx < buf1.shape[0]; idx++)
    {
        arr.push_back(ptr1[idx]);
    }
    std::vector<std::vector<ll>> a(nhashes, std::vector<ll>(width, 0));
    std::vector<std::vector<int>> rnd(nhashes, std::vector<int>(arr.size(), 0));
    std::uniform_int_distribution<> width_distrib(0, width - 1);
    for (int i = 0; i < nhashes; i++)
    {
        for (int j = 0; j < arr.size(); j++)
        {
            rnd[i][j] = width_distrib(gen);
        }
    }

    for (int i = 0; i < nhashes; i++)
    {
        for (int j = 0; j < width; j++)
        {
            a[i][j] = 0LL;
        }
    }

    for (int i = 0; i < nhashes; i++)
    {
        for (int j = 0; j < arr.size(); j++)
        {
            a[i][rnd[i][j]] += arr[j];
        }
    }

    auto res = py::array_t<ll>(buf1.size);

    py::buffer_info buf3 = res.request();
    ll *ptr3 = static_cast<ll *>(buf3.ptr);

    for (int i = 0; i < buf1.shape[0]; i++)
    {
        ll z = 1e18;
        for (int j = 0; j < nhashes; j++)
        {
            z = std::min(z, a[j][rnd[j][i]]);
        }
        ptr3[i] = z;
    }
    return res;
}

double median(std::vector<ll> v)
{
    sort(v.begin(), v.end());
    int sz = (int)v.size();
    return ((double)v[(sz - 1) / 2] + v[(sz) / 2]) / 2.0;
}

py::array_t<double> count_sketch_preds(int nhashes, py::array_t<ll> np_input, ll width, int seed)
{
    //returns the predictions from a count-min sketch
    py::buffer_info buf1 = np_input.request();
    std::mt19937 gen(seed);
    ll *ptr1 = static_cast<ll *>(buf1.ptr);

    std::vector<ll> arr;
    for (size_t idx = 0; idx < buf1.shape[0]; idx++)
    {
        arr.push_back(ptr1[idx]);
    }
    std::vector<std::vector<ll>> a(nhashes, std::vector<ll>(width, 0));
    std::vector<std::vector<int>> rnd(nhashes, std::vector<int>(arr.size(), 0));
    std::vector<std::vector<int>> rnds(nhashes, std::vector<int>(arr.size(), 0));
    std::uniform_int_distribution<> width_distrib(0, width - 1);
    std::uniform_int_distribution<> coin(0, 1);

    for (int i = 0; i < nhashes; i++)
    {
        for (int j = 0; j < arr.size(); j++)
        {
            rnd[i][j] = width_distrib(gen);
        }
    }
    for (int i = 0; i < nhashes; i++)
    {
        for (int j = 0; j < arr.size(); j++)
        {
            rnds[i][j] = coin(gen);
        }
    }

    for (int i = 0; i < nhashes; i++)
    {
        for (int j = 0; j < width; j++)
        {
            a[i][j] = 0LL;
        }
    }

    for (int i = 0; i < nhashes; i++)
    {
        for (int j = 0; j < arr.size(); j++)
        {
            a[i][rnd[i][j]] += (2 * rnds[i][j] - 1) * arr[j];
        }
    }

    auto res = py::array_t<double>(buf1.size);

    py::buffer_info buf3 = res.request();
    double *ptr3 = static_cast<double *>(buf3.ptr);
    std::vector<std::vector<ll>> results(nhashes, std::vector<ll>(arr.size(), 0));
    for (int i = 0; i < nhashes; i++)
    {
        for (int j = 0; j < buf1.shape[0]; j++)
        {
            if (rnds[i][j])
            {
                results[i][j] = a[i][rnd[i][j]];
            }
            else
            {
                results[i][j] = -a[i][rnd[i][j]];
            }
        }
    }
    for (int j = 0; j < buf1.shape[0]; j++)
    {
        std::vector<ll> v(nhashes);
        for (int i = 0; i < nhashes; i++)
        {
            v[i] = results[i][j];
        }
        ptr3[j] = median(v);
    }
    return res;
}

PYBIND11_MODULE(sketches, m)
{
    m.doc() = "Count-Min & Count-Sketch Python Bindings"; // optional module docstring
    m.def("cm_sketch_preds", &cm_sketch_preds, "Count-Min Sketch Predictions");
    m.def("count_sketch_preds", &count_sketch_preds, "Count-Sketch Predictions");
}

#pragma once

#include<iostream>
#include<random>
#include<vector>
#include<initializer_list>
#include<cmath>

#include"../LinearAlgebra/matrix.h"

namespace MLP
{
	using LinearAlgebra::Matrix;

	template<typename T>
	T sigmoid(T x)
	{
		return 1 / (1 + std::exp(-x));
	}
	template<typename T>
	T D_sigmoid(T x)
	{
		return sigmoid(x) * (1 - sigmoid(x));
	}

	template<typename T>
	T relu(T x)
	{
		return std::max(static_cast<T>(0), x);
	}
	template<typename T>
	T D_relu(T x)
	{
		return x > 0 ? static_cast<T>(1) : static_cast<T>(0);
	}



	//class layer
	//{
	//public:
	//	layer() = delete;
	//	~layer() = default;

	//private:

	//};

	template<typename T>
	class mlp
	{
	public:
		using func_type = T(*)(T);

		mlp() = delete;
		mlp(std::initializer_list<int> layer_sizes, std::initializer_list<func_type> activation_functions);
		~mlp() = default;

	private:
		std::vector<Matrix<T>> weights;
		std::vector<Matrix<T>> biases;
		std::vector<func_type> activation_functions;
		std::vector<Matrix<T>> layers;

	};

	template<typename T>
	inline mlp<T>::mlp(std::initializer_list<int> layer_sizes, std::initializer_list<func_type> activation_funcs)
	{
		if (layer_sizes.size() < 2)
		{
			throw std::invalid_argument("MLP must have at least 2 layers");
		}
		if (activation_funcs.size() != layer_sizes.size() - 1)
		{
			throw std::invalid_argument("Number of activation functions must be equal to number of layers - 1");
		}

		for(std::size_t i = 0; i < layer_sizes.size() - 1; ++i)
		{
			if (layer_sizes.begin()[i] <= 0 || layer_sizes.begin()[i + 1] <= 0)
			{
				throw std::invalid_argument("Layer sizes must be positive integers");
			}

			weights.push_back(Matrix<T>(layer_sizes.begin()[i + 1], layer_sizes.begin()[i]));
			biases.push_back(Matrix<T>(layer_sizes.begin()[i + 1], 1));
			layers.push_back(Matrix<T>(layer_sizes.begin()[i], 1));
		}

		layers.push_back(Matrix<T>(layer_sizes.begin()[layer_sizes.size() - 1], 1));
		activation_functions = std::vector<func_type>(activation_funcs);
	}

}

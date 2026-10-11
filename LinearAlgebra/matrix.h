#pragma once

#include<vector>

namespace LinearAlgebra
{
	template<typename T>
	class Matrix
	{
	public:
		using func_type = T(*)(T);

		Matrix() = delete;
		Matrix(size_t rows, size_t cols) :
			rows_(rows), cols_(cols), data_(rows* cols, 0) {}
		~Matrix() = default;

	private:
		size_t rows_, cols_;
		std::vector<T> data_;

	};
}
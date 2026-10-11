#pragma once

#include<vector>
#include<iterator>

namespace LinearAlgebra
{
	template<typename T>
	class Matrix
	{
	public:
		using func_type = T(*)(T);

		Matrix() = delete;
		Matrix(size_t rows, size_t cols) :
			rows_(rows), cols_(cols), data_(rows* cols, 0), size_(rows* cols) {}
		Matrix(const Matrix& other) = default;
		~Matrix() = default;

		T& operator()(size_t row, size_t col) { return data_[get_index(row, col)]; }
		const T& operator()(size_t row, size_t col) const { return data_[get_index(row, col)]; }

		Matrix calculate(func_type func) const;

	private:
		size_t rows_, cols_;
		size_t size_;
		std::vector<T> data_;

		size_t get_index(size_t row, size_t col) const
		{
			if (row >= rows_ || col >= cols_)
			{
				throw std::out_of_range("Matrix index out of bounds");
			}
			return row * cols_ + col;
		}

	};

	template<typename T>
	inline Matrix<T> Matrix<T>::calculate(func_type func) const
	{
		Matrix<T> result(rows_, cols_);

		for (size_t i = 0; i < this->size_; ++i)
		{
			result.data_[i] = func(this->data_[i]);
		}
		return result;
	}

}
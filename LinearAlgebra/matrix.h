#pragma once

#include<vector>
#include<iterator>
#include<stdexcept>

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
		Matrix(Matrix&& other) noexcept = default;
		~Matrix() = default;

		T& operator()(size_t row, size_t col) { return data_[get_index(row, col)]; }
		const T& operator()(size_t row, size_t col) const { return data_[get_index(row, col)]; }

		Matrix<T> calculate(func_type func) const;

		Matrix<T> operator+(const Matrix<T>& other);
		Matrix<T> operator-();
		Matrix<T> operator-(const Matrix<T>& other) { return *this + (-other); }
		Matrix<T> operator*(const T& scalar);
		Matrix<T> operator*(const Matrix<T>& other);
		Matrix<T>& operator+=(const Matrix<T>& other) { *this = *this + other; return *this; }
		Matrix<T>& operator-=(const Matrix<T>& other) { *this = *this - other; return *this; }
		Matrix<T>& operator*=(const T& scalar) { *this = *this * scalar; return *this; }
		Matrix<T>& operator*=(const Matrix<T>& other) { *this = *this * other; return *this; }

	private:
		size_t rows_, cols_;
		size_t size_;//total size of the matrix
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

	template<typename T>
	inline Matrix<T> Matrix<T>::operator+(const Matrix& other)
	{
		if (this->rows_ != other.rows_ || this->cols_ != other.cols_)
		{
			throw std::invalid_argument("Incompatible matrix dimensions");
		}

		Matrix<T> result(rows_, cols_);

		for (size_t i = 0; i < this->size_; ++i)
		{
			result.data_[i] = this->data_[i] + other.data_[i];
		}
		return result;
	}

	template<typename T>
	inline Matrix<T> Matrix<T>::operator-()
	{
		Matrix<T> result(rows_, cols_);
		for (size_t i = 0; i < this->size_; ++i)
		{
			result.data_[i] = -this->data_[i];
		}
		return result;
	}

	template<typename T>
	inline Matrix<T> Matrix<T>::operator*(const T& scalar)
	{
		Matrix<T> result(rows_, cols_);
		for (size_t i = 0; i < this->size_; ++i)
		{
			result.data_[i] = this->data_[i] * scalar;
		}
		return result;
	}

	template<typename T>
	Matrix<T> Matrix<T>::operator*(const Matrix<T>& other)
	{
		if (this->cols_ != other.rows_)
		{
			throw std::invalid_argument("Incompatible matrix dimensions");
		}

		Matrix<T> result(this->rows_, other.cols_);

		for (size_t i = 0; i < this->rows_; ++i)
		{
			for (size_t j = 0; j < other.cols_; ++j)
			{
				T sum = T();
				for (size_t k = 0; k < this->cols_; ++k)
				{
					sum += (*this)(i, k) * other(k, j);
				}
				result(i, j) = sum;
			}
		}
		return result;
	}

}
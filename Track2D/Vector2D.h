/*
 *     Copyright (C) 2013-2018 Sumandeep Banerjee
 * 
 *     This program is free software: you can redistribute it and/or modify
 *     it under the terms of the GNU Lesser General Public License as published by
 *     the Free Software Foundation, either version 3 of the License, or
 *     (at your option) any later version.
 * 
 *     This program is distributed in the hope that it will be useful,
 *     but WITHOUT ANY WARRANTY; without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU General Public License for more details.
 * 
 *     You should have received a copy of the GNU Lesser General Public License
 *     along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

/* 
 * File:   
 * Author: sumandeep
 * Email:  sumandeep.banerjee@gmail.com
*/

#pragma once

#include "opencv2\opencv.hpp"

class CVector2D
{
protected:
	double	m_x;
	double	m_y;

public:
	CVector2D();
	CVector2D( CVector2D &vec );
	CVector2D( double x, double y );
	CVector2D( cv::Point2f &point2f );
	CVector2D( cv::Point2d &point2d );

	double X();
	double Y();

    // binary operators
    CVector2D operator + (const CVector2D& vector) const;
    CVector2D operator - (const CVector2D& vector) const;
    CVector2D operator * (double scalar) const;
    CVector2D operator / (double scalar) const;
    friend CVector2D operator * (double scalar, const CVector2D& vector);

#if 0
	// conditional operators
 	bool isValid () const;
	bool operator == (const C3DVector &vector) const;
    bool operator != (const C3DVector &vector) const;
	friend bool IsVectorParallel (const C3DVector &cV1, const C3DVector &cV2);

	// vector operations
	double Magnitude () const;
	C3DVector Direction ()  const;
	void Normalize ();
	friend double VectorDot (const C3DVector &a, const C3DVector &b);
	friend C3DVector VectorCross (const C3DVector &a, const C3DVector &b);
	friend double VectorAngle (const C3DVector &a, const C3DVector &b);

	// transformations
	friend C3DVector VectorRotate (const C3DVector &vector, const double rotate [3][3]);
	friend C3DVector VectorTranslate (const C3DVector &vector, const double translate [3]);
	friend C3DVector VectorScale (const C3DVector &vector, const double scale [3]);
#endif

	friend CVector2D vectorAverage( std::vector<CVector2D> &vecArray );
};

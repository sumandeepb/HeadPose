#include "Vector2D.h"

CVector2D::CVector2D()
{
	m_x = 0.0;
	m_y = 0.0;
}

CVector2D::CVector2D( CVector2D &vec )
{
	m_x = vec.m_x;
	m_y = vec.m_y;
}

CVector2D::CVector2D( double x, double y )
{
	m_x = x;
	m_y = y;
}

CVector2D::CVector2D( cv::Point2f &point2f )
{
	m_x = (double)point2f.x;
	m_y = (double)point2f.y;
}

CVector2D::CVector2D( cv::Point2d &point2d )
{
	m_x = point2d.x;
	m_y = point2d.y;
}

double CVector2D::X()
{
	return m_x;
}

double CVector2D::Y()
{
	return m_y;
}

// binary operators
CVector2D CVector2D::operator + (const CVector2D& vector) const
{
	CVector2D cTempVector ((*this).m_x + vector.m_x, (*this).m_y + vector.m_y);
	return cTempVector;
}

CVector2D CVector2D::operator - (const CVector2D& vector) const
{
	CVector2D cTempVector ((*this).m_x - vector.m_x, (*this).m_y - vector.m_y);
	return cTempVector;
}

CVector2D CVector2D::operator * (double scalar) const
{
	CVector2D cTempVector ((*this).m_x * scalar,	(*this).m_y * scalar);
	return cTempVector;
}

CVector2D CVector2D::operator / (double scalar) const
{
	CVector2D cTempVector ((*this).m_x / scalar, (*this).m_y / scalar);
	return cTempVector;
}

CVector2D operator * (double scalar, const CVector2D& vector)
{
	return (vector * scalar);
}

#if 0
// conditional operators
bool CVector2D::isValid () const
{
	return !(*this == CVector2D());
}

bool C3DVector::operator == (const C3DVector& vector) const
{
	return (fabs ((*this).m_dX - vector.m_dX) < EPSILON && fabs ((*this).m_dY - vector.m_dY) < EPSILON && fabs ((*this).m_dZ - vector.m_dZ) < EPSILON);
}

bool C3DVector::operator != (const C3DVector& vector) const
{
	return (!((*this) == vector));
}


bool IsVectorParallel (const C3DVector &cV1, const C3DVector &cV2)
{
	if (cV1.Direction () == cV2.Direction () || cV1.Direction () == -1 * cV2.Direction ())
	{
		return true;
	}
	return false;
}

// vector operations
double C3DVector::Magnitude () const
{
	return (sqrt (m_dX * m_dX + m_dY * m_dY + m_dZ * m_dZ));
}

C3DVector C3DVector::Direction () const
{
	double		dMagnitude = C3DVector::Magnitude ();
	C3DVector	cTempVector (m_dX / dMagnitude, m_dY / dMagnitude, m_dZ / dMagnitude);
	return cTempVector;
}

void C3DVector::Normalize ()
{
	(*this) = C3DVector::Direction ();
}

double VectorDot (const C3DVector& a, const C3DVector& b)
{
	return (a.m_dX * b.m_dX + a.m_dY * b.m_dY + a.m_dZ * b.m_dZ);
}

C3DVector VectorCross (const C3DVector& a, const C3DVector& b)
{
	C3DVector cTempVector (a.m_dY * b.m_dZ - b.m_dY * a.m_dZ, b.m_dX * a.m_dZ - a.m_dX * b.m_dZ, a.m_dX * b.m_dY - b.m_dX * a.m_dY);
	return cTempVector;
}

double VectorAngle (const C3DVector& a, const C3DVector& b)
{
	return acos ((a.m_dX * b.m_dX + a.m_dY * b.m_dY + a.m_dZ * b.m_dZ) / 
		((sqrt (a.m_dX * a.m_dX + a.m_dY * a.m_dY + a.m_dZ * a.m_dZ)) * (sqrt (b.m_dX * b.m_dX + b.m_dY * b.m_dY + b.m_dZ * b.m_dZ))));
}

// transformations
C3DVector VectorRotate (const C3DVector& vector, const double r[3][3])
{
	C3DVector cTempVector;
	cTempVector.m_dX = r[0][0] * vector.m_dX + r[0][1] * vector.m_dY + r[0][2] * vector.m_dZ;
	cTempVector.m_dY = r[1][0] * vector.m_dX + r[1][1] * vector.m_dY + r[1][2] * vector.m_dZ;
	cTempVector.m_dZ = r[2][0] * vector.m_dX + r[2][1] * vector.m_dY + r[2][2] * vector.m_dZ;
	return cTempVector;
}

C3DVector VectorTranslate (const C3DVector& vector, const double t[3])
{
	C3DVector cTempVector (vector.m_dX + t[0], vector.m_dY + t[1], vector.m_dZ + t[2]);
	return cTempVector;
}

C3DVector VectorScale (const C3DVector& vector, const double s[3])
{
	C3DVector cTempVector (vector.m_dX * s[0], vector.m_dY * s[1], vector.m_dZ * s[2]);
	return cTempVector;
}


#endif

CVector2D vectorAverage( std::vector<CVector2D> &vecArray )
{
	CVector2D	avgVec;
	int			nCount;
	std::vector<CVector2D>::iterator it;

	for( it = vecArray.begin(), nCount = 0; it != vecArray.end(); it++, nCount++ )
	{
		avgVec = avgVec + *it;
	}

	avgVec = avgVec / double(nCount);

	return avgVec;
}

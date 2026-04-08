import ezdxf
import geopandas as gpd
from shapely.geometry import LineString, Point, Polygon

class DXFFile:
    def __init__(self, dxf_path, crs):
        self.dxf_path = dxf_path
        self.crs = crs

        self.load()

    def load(self):
        """
        Load DXF file
        """
        doc = ezdxf.readfile(self.dxf_path)
        msp = doc.modelspace()  # Main drawing content
        self.msp = msp
        print("DXF file successfully loaded.")
    
    def describe(self):
        """
        List ALL unique entity types in this file
        """
        entity_types = set()
        for entity in self.msp:
            entity_types.add(entity.dxftype())
            
        print("Entities in this DXF:")
        for etype in sorted(entity_types):
            count = len(self.msp.query(etype))
            print(f"  {etype}: {count}")

    def extract_features(self, feature_type_list):
        """
        """
        output = {}
        try:
            if 'LINE' in feature_type_list:
                features = []
                for line in self.msp.query('LINE'):
                    start = tuple(line.dxf.start)[:2]  # Vec3 → (x, y)
                    end = tuple(line.dxf.end)[:2]      # Vec3 → (x, y)
                    geom = LineString([start, end])
                    features.append({'geometry': geom, 'layer': line.dxf.layer})
                lines_gdf = gpd.GeoDataFrame(features, crs=self.crs)
                lines_gdf['length'] = lines_gdf.geometry.length
                output["LINE"] = lines_gdf

            if 'LWPOLYLINE' in feature_type_list:
                features = []
                for pline in self.msp.query('LWPOLYLINE'):
                    # Get ALL vertices as (x, y) tuples
                    vertices = [(v[0], v[1]) for v in pline.vertices()]  # List of (x,y)
                    # Convert to LineString (or MultiLineString if closed/open)
                    if len(vertices) > 1:
                        geom = LineString(vertices)
                        features.append({'geometry': geom, 'layer': pline.dxf.layer})
                polylines_gdf = gpd.GeoDataFrame(features, crs=self.crs)
                polylines_gdf['length'] = polylines_gdf.geometry.length
                output["LWPOLYLINE"] = polylines_gdf

            if 'POINT' in feature_type_list:
                points_features = []
                for point in self.msp.query('POINT'):
                    # Get x, y coordinates from POINT entity
                    x = point.dxf.location.x    # Single point location
                    y = point.dxf.location.y
                    geom = Point(x, y)
                    points_features.append({'geometry': geom, 'layer': point.dxf.layer})
                points_gdf = gpd.GeoDataFrame(points_features, crs=self.crs)
                output["POINT"] = points_gdf

            if 'HATCH' in feature_type_list:
                hatch_features = []
                for hatch in self.msp.query('HATCH'):
                    # Get boundary paths as polygons
                    for path in hatch.paths:
                        if path.type == 1:  # Polyline path
                            vertices = [(v[0], v[1]) for v in path.vertices]
                            if len(vertices) > 2:  # Valid polygon
                                geom = Polygon(vertices)
                                hatch_features.append({
                                    'geometry': geom, 
                                    'layer': hatch.dxf.layer,
                                    'pattern': hatch.dxf.pattern_name  # SOLID, ANSI31, etc.
                                })
                hatches_gdf = gpd.GeoDataFrame(hatch_features, crs=self.crs)
                output["HATCH"] = hatches_gdf

            if 'CIRCLE' in feature_type_list:
                # Circles → Points (centers)
                circle_features = []
                for circle in self.msp.query('CIRCLE'):
                    geom = Point(circle.dxf.center.x, circle.dxf.center.y)
                    circle_features.append({
                        'geometry': geom, 
                        'layer': circle.dxf.layer,
                        'radius': circle.dxf.radius
                    })
                circles_gdf = gpd.GeoDataFrame(circle_features, crs=self.crs)
                output["CIRCLE"] = circles_gdf

            if ('TEXT' in feature_type_list) or ('MTEXT' in feature_type_list):
                text_features = []
                for text_entity in self.msp.query('TEXT MTEXT'):  # Both types
                    if text_entity.dxftype() == 'TEXT':
                        geom = Point(text_entity.dxf.insert.x, text_entity.dxf.insert.y)
                        text_content = text_entity.dxf.text
                        height = text_entity.dxf.height
                    else:  # MTEXT
                        geom = Point(text_entity.dxf.insert.x, text_entity.dxf.insert.y)
                        text_content = text_entity.dxf.text
                        height = text_entity.dxf.char_height
                    
                    text_features.append({
                        'geometry': geom,
                        'layer': text_entity.dxf.layer,
                        'text': text_content,
                        'height': height
                    })
                texts_gdf = gpd.GeoDataFrame(text_features, crs=self.crs)
                output["TEXT"] = texts_gdf
            
            # for label, gdf in output.items():
            #     setattr(self, label, gdf)  # Dynamic attribute: self.label = gdf

            if not hasattr(self, 'features'):
                self.features = {}
                print("Initialized self.features dict")

            for label, gdf in output.items():
                old_count = len(self.features.get(label, []))  # 0 if missing
                self.features[label] = gdf
                new_count = len(gdf)
                print(f"Updated {label}: {old_count} → {new_count} features")

        except Exception as e:
            raise

    def to_geodataframes(self):
        """Return extracted features as a dict of GeoDataFrames."""
        if not hasattr(self, "features"):
            raise ValueError(
                "No features extracted yet. Run extract_features(...) first."
            )
        return self.features

    def to_geodataframe(self, feature_types=None):
        """
        Return one merged GeoDataFrame with all requested feature types.
        """
        if not hasattr(self, "features"):
            raise ValueError(
                "No features extracted yet. Run extract_features(...) first."
            )

        if feature_types is None:
            feature_types = list(self.features.keys())

        frames = []
        for feature_type in feature_types:
            if feature_type in self.features:
                gdf = self.features[feature_type].copy()
                gdf["feature_type"] = feature_type
                frames.append(gdf)

        if not frames:
            return gpd.GeoDataFrame(geometry=[], crs=self.crs)

        merged = pd.concat(frames, ignore_index=True)
        return gpd.GeoDataFrame(merged, geometry="geometry", crs=self.crs)

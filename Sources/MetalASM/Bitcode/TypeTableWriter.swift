/// Writes the TYPE_BLOCK in LLVM bitcode format.
///
/// Each type in the enumerator's type table is serialized as a record.
/// Type references use the table index.
final class TypeTableWriter {
    // TYPE_BLOCK record codes
    static let numEntryCode: UInt64 = 1
    static let voidCode: UInt64 = 2
    static let floatCode: UInt64 = 3
    static let doubleCode: UInt64 = 4
    static let labelCode: UInt64 = 5
    static let opaqueCode: UInt64 = 6
    static let integerCode: UInt64 = 7
    static let pointerCode: UInt64 = 8
    static let functionOldCode: UInt64 = 9  // function type (old style)
    static let halfCode: UInt64 = 10
    static let arrayCode: UInt64 = 11
    static let vectorCode: UInt64 = 12
    static let metadataCode: UInt64 = 16
    static let structAnonCode: UInt64 = 18
    static let structNameCode: UInt64 = 19
    static let structNamedCode: UInt64 = 20
    static let functionCode: UInt64 = 21  // function type (new style)
    static let tokenCode: UInt64 = 22
    static let bfloatCode: UInt64 = 23
    /// TYPE_CODE_OPAQUE_POINTER (LLVM 17 opaque pointer, "ptr").
    static let opaquePtrCode: UInt64 = 25

    /// When true, emit opaquePointer as typed pointer (float*).
    /// Set by BitcodeWriter when MMA intrinsics are detected.
    static var emitOpaqueAsTyped = false

    /// TYPE_BLOCK ID.
    static let blockID: UInt64 = 17

    /// Write the TYPE_BLOCK.
    static func write(to writer: BitstreamWriter, enumerator: ValueEnumerator) {
        writer.enterSubblock(blockID: blockID, abbrevLen: 4)

        // NUMENTRY record: total number of type entries
        writer.emitUnabbrevRecord(code: numEntryCode, operands: [UInt64(enumerator.types.count)])

        for type in enumerator.types {
            writeType(type, to: writer, enumerator: enumerator)
        }

        writer.exitBlock()
    }

    /// Write a single type record.
    private static func writeType(_ type: IRType, to writer: BitstreamWriter, enumerator: ValueEnumerator) {
        switch type {
        case .void:
            writer.emitUnabbrevRecord(code: voidCode, operands: [])

        case .int(let bits):
            writer.emitUnabbrevRecord(code: integerCode, operands: [UInt64(bits)])

        case .float16:
            writer.emitUnabbrevRecord(code: halfCode, operands: [])

        case .bfloat16:
            writer.emitUnabbrevRecord(code: bfloatCode, operands: [])

        case .float32:
            writer.emitUnabbrevRecord(code: floatCode, operands: [])

        case .float64:
            writer.emitUnabbrevRecord(code: doubleCode, operands: [])

        case .label:
            writer.emitUnabbrevRecord(code: labelCode, operands: [])

        case .metadata:
            writer.emitUnabbrevRecord(code: metadataCode, operands: [])

        case .token:
            writer.emitUnabbrevRecord(code: tokenCode, operands: [])

        case .opaquePointer(let addrSpace):
            if TypeTableWriter.emitOpaqueAsTyped {
                // Emit as typed pointer (float*) — required when MMA intrinsics are present
                let floatIdx = enumerator.typeIndex(.float32)
                writer.emitUnabbrevRecord(code: pointerCode, operands: [
                    UInt64(floatIdx), UInt64(addrSpace)
                ])
            } else {
                // Standard opaque pointer encoding
                writer.emitUnabbrevRecord(code: opaquePtrCode, operands: [UInt64(addrSpace)])
            }

        case .pointer(let pointee, let addrSpace):
            // Typed pointer (legacy AIR style) → POINTER: [pointee_type_id, address_space]
            writer.emitUnabbrevRecord(code: pointerCode, operands: [
                UInt64(enumerator.typeIndex(pointee)),
                UInt64(addrSpace)
            ])

        case .array(let elem, let count):
            // ARRAY: [count, element_type_id]
            writer.emitUnabbrevRecord(code: arrayCode, operands: [
                UInt64(count),
                UInt64(enumerator.typeIndex(elem))
            ])

        case .vector(let elem, let count):
            // VECTOR: [count, element_type_id]
            writer.emitUnabbrevRecord(code: vectorCode, operands: [
                UInt64(count),
                UInt64(enumerator.typeIndex(elem))
            ])

        case .function(let ret, let params, let isVarArg):
            // FUNCTION: [isVarArg, retType, ...paramTypes]
            var operands: [UInt64] = [isVarArg ? 1 : 0, UInt64(enumerator.typeIndex(ret))]
            for p in params {
                operands.append(UInt64(enumerator.typeIndex(p)))
            }
            writer.emitUnabbrevRecord(code: functionCode, operands: operands)

        case .structure(let name, let elems, let isPacked):
            if let name = name {
                // Named struct: first emit name, then body
                writer.emitUnabbrevStringRecord(code: structNameCode, name)
                // STRUCT_NAMED: [isPacked, ...elementTypes]
                var operands: [UInt64] = [isPacked ? 1 : 0]
                for elem in elems {
                    operands.append(UInt64(enumerator.typeIndex(elem)))
                }
                writer.emitUnabbrevRecord(code: structNamedCode, operands: operands)
            } else {
                // Anonymous struct: STRUCT_ANON: [isPacked, ...elementTypes]
                var operands: [UInt64] = [isPacked ? 1 : 0]
                for elem in elems {
                    operands.append(UInt64(enumerator.typeIndex(elem)))
                }
                writer.emitUnabbrevRecord(code: structAnonCode, operands: operands)
            }

        case .opaque(let name):
            // Opaque struct: emit name first, then OPAQUE [ispacked]
            writer.emitUnabbrevStringRecord(code: structNameCode, name)
            writer.emitUnabbrevRecord(code: opaqueCode, operands: [0])  // ispacked=0
        }
    }
}
